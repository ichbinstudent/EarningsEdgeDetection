#!/usr/bin/env python3
"""
Unified Telegram bot for the Earnings Edge trading scanners.

Runs both the US Earnings Calendar scanner and the Eurex Forward Volatility
scanner on configurable cron schedules, pushing results to subscribers via
Telegram. Also supports on-demand /run and keyboard-driven interaction.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Set

from dotenv import load_dotenv

# Ensure project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from earnings_edge.base import BaseScanner
from earnings_edge.bot_scanner import EarningsCalendarScanner
from earnings_edge.forward_volatility import ForwardVolatilityScanner

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler,
    ContextTypes, MessageHandler, filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("trading_bot")

# ── Keyboard layouts ──────────────────────────────────────────────────────

MAIN_KB = [
    ["📊 Scanners", "📋 My Subscriptions"],
    ["🔄 Run Scanner", "❓ Help"],
    ["🚪 Close Keyboard"],
]
SCANNER_KB = [
    ["📊 List Scanners", "📋 My Subscriptions"],
    ["🔄 Run Scanner", "⬅️ Back to Main"],
]
RUN_KB = [["⬅️ Back to Main"]]


# ── Health endpoint ──────────────────────────────────────────────────────

class _HealthHandler(BaseHTTPRequestHandler):
    """Lightweight HTTP handler returning bot health JSON."""
    _started_at = time.monotonic()

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        uptime = round(time.monotonic() - self._started_at, 1)
        body = json.dumps({"status": "ok", "uptime_secs": uptime}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # suppress request logging


def _start_health_server(port: int = 8502):
    """Run the health HTTP server in a daemon thread."""
    try:
        server = HTTPServer(("127.0.0.1", port), _HealthHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        logger.info("Health endpoint started on http://127.0.0.1:%d/health", port)
    except OSError as exc:
        logger.warning("Could not start health endpoint on port %d: %s", port, exc)


# ── Bot ───────────────────────────────────────────────────────────────────

class TradingBot:
    def __init__(self, token: str):
        self.token = token
        self.subscribers_file = os.path.join(os.path.dirname(__file__), "data", "subscribers.json")
        self.scanners: Dict[str, BaseScanner] = {}
        self.subscribers: Dict[str, Set[int]] = {}
        self.application = None
        self._last_scan_results: dict[tuple[int, str], dict] = {}

        self._load_subscribers()
        self._register(EarningsCalendarScanner())
        self._register(ForwardVolatilityScanner())

        self.scheduler = BlockingScheduler()
        self._scheduler_thread = None

    # ── Registration / persistence ─────────────────────────────────────

    def _register(self, scanner: BaseScanner):
        self.scanners[scanner.name] = scanner
        self.subscribers.setdefault(scanner.name, set())
        logger.info("Registered scanner: %s (schedule: %s)", scanner.name, scanner.schedule)

    def _load_subscribers(self):
        if os.path.exists(self.subscribers_file):
            try:
                with open(self.subscribers_file) as f:
                    self.subscribers = {k: set(v) for k, v in json.load(f).items()}
            except Exception as exc:
                logger.error("Failed to load subscribers: %s", exc)

    def _save_subscribers(self):
        os.makedirs(os.path.dirname(self.subscribers_file), exist_ok=True)
        with open(self.subscribers_file, "w") as f:
            json.dump({k: list(v) for k, v in self.subscribers.items()}, f, indent=2)

    def _subscribe(self, name: str, uid: int) -> bool:
        if name in self.scanners:
            self.subscribers[name].add(uid)
            self._save_subscribers()
            return True
        return False

    def _unsubscribe(self, name: str, uid: int) -> bool:
        if name in self.subscribers:
            self.subscribers[name].discard(uid)
            self._save_subscribers()
            return True
        return False

    def _user_subs(self, uid: int):
        return [n for n, uids in self.subscribers.items() if uid in uids]

    # ── Format helpers ─────────────────────────────────────────────────

    @staticmethod
    def _format_results(embed: dict) -> str:
        title = embed.get("title", "Results")
        fields = embed.get("fields", [])
        msg = f"📊 {title}\n\n"
        for f in fields:
            msg += f"{f['name']}\n{f['value']}\n\n"
        return msg

    @staticmethod
    def _chunk_text(text: str, limit: int = 3800) -> list[str]:
        """Split text into Telegram-safe chunks."""
        if len(text) <= limit:
            return [text]
        chunks = []
        buf = ""
        for para in text.split("\n\n"):
            candidate = f"{buf}\n\n{para}" if buf else para
            if len(candidate) <= limit:
                buf = candidate
                continue
            if buf:
                chunks.append(buf)
                buf = ""
            if len(para) <= limit:
                buf = para
            else:
                start = 0
                while start < len(para):
                    chunks.append(para[start:start + limit])
                    start += limit
        if buf:
            chunks.append(buf)
        return chunks

    @staticmethod
    def _take_only_embed(embed: dict) -> dict:
        fields = embed.get("fields", [])
        take_fields = [
            f for f in fields
            if f.get("name") == "Summary — TAKE trades only" or "→ TAKE" in str(f.get("value", ""))
        ]
        return {**embed, "fields": take_fields or [{"name": "Summary", "value": "No TAKE trades today.", "inline": False}]}

    @staticmethod
    def _full_embed(embed: dict) -> dict:
        return embed

    # ── Command handlers ───────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        kb = ReplyKeyboardMarkup(MAIN_KB, resize_keyboard=True)
        await update.message.reply_text(
            "🚀 *Welcome to the Trading Bot!*\n\n"
            "📈 Get notified about trading opportunities.\n\n"
            "Commands:\n"
            "• `/scanners` — Browse scanners\n"
            "• `/subscriptions` — Manage subscriptions\n"
            "• `/run` — Run a scanner on-demand\n"
            "• `/help` — Detailed help\n",
            reply_markup=kb, parse_mode="Markdown",
        )

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        kb = ReplyKeyboardMarkup(MAIN_KB, resize_keyboard=True)
        lines = [
            "🆘 *Trading Bot Help*\n",
            "📋 *Commands:*",
            "• `/start` — Main menu",
            "• `/scanners` — Browse & subscribe",
            "• `/subscriptions` — View active subs",
            "• `/run` — On-demand scanner execution",
            "• `/help` — This message\n",
            "⏰ *Schedules:*",
        ]
        for name, sc in self.scanners.items():
            lines.append(f"• {name}: {sc.schedule}")
        lines += [
            "\n📱 Use the keyboard at the bottom for quick access!",
        ]
        await update.message.reply_text(
            "\n".join(lines), reply_markup=kb, parse_mode="Markdown",
        )

    async def _cmd_scanners(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        msg = "📊 *Available Scanners*\n\n"
        ikb = []
        for name, sc in self.scanners.items():
            subbed = uid in self.subscribers.get(name, set())
            status = "✅ Subscribed" if subbed else "❌ Not subscribed"
            action = "Unsubscribe" if subbed else "Subscribe"
            cb = f"unsub_{name}" if subbed else f"sub_{name}"
            msg += f"🔹 *{name}*\n  ⏰ {sc.get_schedule_description()}\n  {status}\n\n"
            ikb.append([InlineKeyboardButton(f"{action}: {name}", callback_data=cb)])

        kb = ReplyKeyboardMarkup(SCANNER_KB, resize_keyboard=True)
        await update.message.reply_text(msg, reply_markup=kb, parse_mode="Markdown")
        await update.message.reply_text("💡 *Quick Actions:*", reply_markup=InlineKeyboardMarkup(ikb), parse_mode="Markdown")

    async def _cmd_subscriptions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        subs = self._user_subs(uid)
        kb = ReplyKeyboardMarkup(MAIN_KB, resize_keyboard=True)
        if not subs:
            await update.message.reply_text(
                "📋 *Your Subscriptions*\n\n❌ None active.\nUse /scanners to browse!",
                reply_markup=kb, parse_mode="Markdown",
            )
            return
        msg = "📋 *Active Subscriptions*\n\n"
        ikb = []
        for name in subs:
            sc = self.scanners[name]
            msg += f"🔹 *{name}*\n  ⏰ {sc.get_schedule_description()}\n\n"
            ikb.append([InlineKeyboardButton(f"Unsubscribe: {name}", callback_data=f"unsub_{name}")])
        await update.message.reply_text(msg, reply_markup=kb, parse_mode="Markdown")
        await update.message.reply_text("💡 *Quick Actions:*", reply_markup=InlineKeyboardMarkup(ikb), parse_mode="Markdown")

    async def _cmd_run(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        msg = "🔄 *Run Scanner*\n\nChoose a scanner:\n\n"
        ikb = []
        for name, sc in self.scanners.items():
            msg += f"📈 *{name}*\n  ⏰ {sc.get_schedule_description()}\n\n"
            ikb.append([InlineKeyboardButton(f"🚀 Run {name}", callback_data=f"run_{name}")])
        kb = ReplyKeyboardMarkup(RUN_KB, resize_keyboard=True)
        await update.message.reply_text(msg, reply_markup=kb, parse_mode="Markdown")
        await update.message.reply_text("⚡ *Run:*", reply_markup=InlineKeyboardMarkup(ikb), parse_mode="Markdown")

    # ── Callback handler ───────────────────────────────────────────────

    async def _handle_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        uid = query.from_user.id
        data = query.data

        if data.startswith("sub_"):
            name = data[4:]
            if self._subscribe(name, uid):
                await query.edit_message_text(f"✅ Subscribed to {name}!")
            else:
                await query.edit_message_text(f"❌ Failed to subscribe to *{name}*.", parse_mode="Markdown")

        elif data.startswith("unsub_"):
            name = data[5:]
            if self._unsubscribe(name, uid):
                await query.edit_message_text(f"✅ Unsubscribed from {name}.")
            else:
                await query.edit_message_text(f"❌ Failed to unsubscribe from {name}.")

        elif data.startswith("run_"):
            name = data[4:]
            await query.edit_message_text(f"🚀 Running {name}... please wait.")
            try:
                result = self.scanners[name].scan()
                if result.get("success"):
                    self._last_scan_results[(uid, name)] = result["embed"]
                    text = self._format_results(self._take_only_embed(result["embed"]))
                    preview = text[:3500]
                    if len(text) > 3500:
                        preview += "\n\n… output truncated; full result sent separately."
                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton("Show all trades", callback_data=f"showall_{name}")],
                    ])
                    await query.edit_message_text(preview, reply_markup=keyboard)
                    return
                else:
                    text = f"❌ {name} failed: {result.get('error', 'Unknown')}"
                await query.edit_message_text(text)
            except Exception as exc:
                await query.edit_message_text(f"❌ Error: {exc}")

        elif data.startswith("showall_"):
            name = data[8:]
            embed = self._last_scan_results.get((uid, name))
            if not embed:
                await query.edit_message_text("❌ No cached scan found. Run the scanner again.")
                return
            text = self._format_results(self._full_embed(embed))
            preview = text[:3500]
            if len(text) > 3500:
                preview += "\n\n… output truncated; full result sent separately."
            await query.edit_message_text(preview)
            for part in self._chunk_text(text):
                await self.application.bot.send_message(chat_id=uid, text=part)

    # ── Keyboard handler ───────────────────────────────────────────────

    async def _handle_keyboard(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = update.message.text
        if text == "📊 Scanners" or text == "📊 List Scanners":
            await self._cmd_scanners(update, ctx)
        elif text == "📋 My Subscriptions":
            await self._cmd_subscriptions(update, ctx)
        elif text == "🔄 Run Scanner":
            await self._cmd_run(update, ctx)
        elif text == "❓ Help":
            await self._cmd_help(update, ctx)
        elif text == "🚪 Close Keyboard":
            await update.message.reply_text(
                "⌨️ Keyboard closed. /start to bring it back.",
                reply_markup=ReplyKeyboardRemove(),
            )
        elif text == "⬅️ Back to Main":
            kb = ReplyKeyboardMarkup(MAIN_KB, resize_keyboard=True)
            await update.message.reply_text("🏠 Main Menu", reply_markup=kb, parse_mode="Markdown")

    # ── Scheduled scanner run ──────────────────────────────────────────

    async def _run_and_push(self, scanner_name: str):
        if scanner_name not in self.scanners:
            return
        sc = self.scanners[scanner_name]
        logger.info("Scheduled run: %s", scanner_name)
        try:
            result = sc.scan()
            if not result.get("success"):
                logger.error("Scanner %s failed: %s", scanner_name, result.get("error"))
                return
            subs = self.subscribers.get(scanner_name, set())
            if not subs:
                return
            text = self._format_results(result["embed"])
            parts = self._chunk_text(text)
            for uid in subs:
                try:
                    for part in parts:
                        await self.application.bot.send_message(chat_id=uid, text=part, parse_mode="Markdown")
                except Exception as exc:
                    logger.error("Push to %d failed: %s", uid, exc)
        except Exception as exc:
            logger.exception("Scheduled run error for %s", scanner_name)

    def _run_sync(self, scanner_name: str):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_and_push(scanner_name))
        finally:
            loop.close()

    def _setup_scheduler(self):
        tz = pytz.timezone("Europe/Berlin")
        for name, sc in self.scanners.items():
            try:
                trigger = CronTrigger.from_crontab(sc.schedule, timezone=tz)
                self.scheduler.add_job(
                    self._run_sync, trigger=trigger, args=[name],
                    id=f"scanner_{name}", name=f"Run {name}",
                )
                logger.info("Scheduled %s: %s (Berlin TZ)", name, sc.schedule)
            except Exception as exc:
                logger.error("Failed to schedule %s: %s", name, exc)

    # ── Main entry ─────────────────────────────────────────────────────

    def run(self):
        self.application = Application.builder().token(self.token).build()

        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("help", self._cmd_help))
        self.application.add_handler(CommandHandler("scanners", self._cmd_scanners))
        self.application.add_handler(CommandHandler("subscriptions", self._cmd_subscriptions))
        self.application.add_handler(CommandHandler("run", self._cmd_run))
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_keyboard)
        )

        self._setup_scheduler()
        self._scheduler_thread = threading.Thread(target=self.scheduler.start, daemon=True)
        self._scheduler_thread.start()

        _start_health_server()

        logger.info("Trading bot starting (polling mode)...")
        try:
            self.application.run_polling()
        except KeyboardInterrupt:
            pass
        finally:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=True)
            logger.info("Bot shut down.")


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)
    bot = TradingBot(token)
    bot.run()


if __name__ == "__main__":
    main()
