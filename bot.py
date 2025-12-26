

#!/usr/bin/env python3
"""
Main trading bot that manages Telegram subscribers and schedules scanners.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Set

from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import threading

from scanner_base import BaseScanner
from earnings_scanner import EarningsCalendarScanner
from forward_volatility_scanner import ForwardVolatilityScanner

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Keyboard layouts
MAIN_KEYBOARD = [
    ['📊 Scanners', '📋 My Subscriptions'],
    ['🔄 Run Scanner', '❓ Help'],
    ['🚪 Close Keyboard']
]

SCANNER_KEYBOARD = [
    ['📊 List Scanners', '📋 My Subscriptions'],
    ['🔄 Run Scanner', '⬅️ Back to Main']
]

RUN_SCANNER_KEYBOARD = [
    ['⬅️ Back to Main']
]
from dotenv import load_dotenv

load_dotenv()


class TradingBot:
    """
    Main trading bot that handles Telegram interactions and scanner scheduling.
    """

    def __init__(self, token: str):
        self.token = token
        self.subscribers_file = "subscribers.json"
        self.scanners: Dict[str, BaseScanner] = {}
        self.subscribers: Dict[str, Set[int]] = {}  # scanner_name -> set of user_ids

        # Load subscribers from file
        self.load_subscribers()

        # Initialize scanners
        self.register_scanner(EarningsCalendarScanner())
        self.register_scanner(ForwardVolatilityScanner())

        # Initialize scheduler
        self.scheduler = BlockingScheduler()
        self.scheduler_thread = None

    def register_scanner(self, scanner: BaseScanner):
        """Register a scanner with the bot."""
        self.scanners[scanner.name] = scanner
        if scanner.name not in self.subscribers:
            self.subscribers[scanner.name] = set()
        logger.info(f"Registered scanner: {scanner.name} with schedule: {scanner.schedule}")

    def load_subscribers(self):
        """Load subscribers from JSON file."""
        if os.path.exists(self.subscribers_file):
            try:
                with open(self.subscribers_file, 'r') as f:
                    data = json.load(f)
                    # Convert lists back to sets
                    self.subscribers = {k: set(v) for k, v in data.items()}
                logger.info(f"Loaded subscribers: {self.subscribers}")
            except Exception as e:
                logger.error(f"Error loading subscribers: {e}")
                self.subscribers = {}
        else:
            self.subscribers = {}

    def save_subscribers(self):
        """Save subscribers to JSON file."""
        try:
            # Convert sets to lists for JSON serialization
            data = {k: list(v) for k, v in self.subscribers.items()}
            with open(self.subscribers_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Subscribers saved successfully")
        except Exception as e:
            logger.error(f"Error saving subscribers: {e}")

    def subscribe_user(self, scanner_name: str, user_id: int) -> bool:
        """Subscribe a user to a scanner."""
        if scanner_name in self.scanners:
            self.subscribers[scanner_name].add(user_id)
            self.save_subscribers()
            return True
        return False

    def unsubscribe_user(self, scanner_name: str, user_id: int) -> bool:
        """Unsubscribe a user from a scanner."""
        if scanner_name in self.subscribers:
            self.subscribers[scanner_name].discard(user_id)
            self.save_subscribers()
            return True
        return False

    def get_user_subscriptions(self, user_id: int) -> List[str]:
        """Get list of scanners a user is subscribed to."""
        return [name for name, users in self.subscribers.items() if user_id in users]

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        keyboard = ReplyKeyboardMarkup(MAIN_KEYBOARD, resize_keyboard=True)
        await update.message.reply_text(
            "🚀 *Welcome to the Trading Bot!*\n\n"
            "📈 Get notified about trading opportunities from our scanners.\n\n"
            "Use the buttons below or type commands:\n"
            "• `/scanners` - Browse available scanners\n"
            "• `/subscriptions` - Manage your subscriptions\n"
            "• `/help` - Show detailed help\n\n"
            "_Choose an option to get started!_",
            reply_markup=keyboard,
            parse_mode='Markdown'
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        keyboard = ReplyKeyboardMarkup(MAIN_KEYBOARD, resize_keyboard=True)
        help_text = """
🆘 *Trading Bot Help*

📋 *Available Commands:*
• `/start` - Welcome message and main menu
• `/scanners` - Browse and subscribe to scanners
• `/subscriptions` - View your active subscriptions
• `/run` - Manually execute scanners on demand
• `/help` - Show this help message

💡 *How to Use:*
1. Use `/scanners` to see available trading scanners
2. Click the subscribe buttons to get notifications
3. Check `/subscriptions` to manage your subscriptions
4. Use `/run` to execute scanners manually anytime
5. Receive automatic notifications when scanners run

⏰ *Scanner Schedules:*
• Earnings Calendar: Daily at 16:15 EST (weekdays)

⚡ *Manual Execution:*
• Run scanners on-demand using the 🔄 Run Scanner button
• Get immediate results without waiting for scheduled times
• Useful for testing or getting current market data

📱 *Keyboard Shortcuts:*
Use the persistent keyboard at the bottom for quick access!

❓ *Need More Help?*
Contact the bot administrator if you have questions.
        """.strip()
        
        await update.message.reply_text(
            help_text,
            reply_markup=keyboard,
            parse_mode='Markdown'
        )

    async def scanners_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /scanners command."""
        user_id = update.effective_user.id

        if not self.scanners:
            keyboard = ReplyKeyboardMarkup(MAIN_KEYBOARD, resize_keyboard=True)
            await update.message.reply_text(
                "❌ No scanners available at the moment.",
                reply_markup=keyboard
            )
            return

        message = "📊 *Available Trading Scanners*\n\n"
        inline_keyboard = []

        for name, scanner in self.scanners.items():
            subscribers_count = len(self.subscribers.get(name, set()))
            is_subscribed = user_id in self.subscribers.get(name, set())
            
            status_emoji = "✅" if is_subscribed else "❌"
            action_text = "Unsubscribe" if is_subscribed else "Subscribe"
            callback_data = f"unsubscribe_{name}" if is_subscribed else f"subscribe_{name}"
            
            message += f"� *{name}*\n"
            message += f"   ⏰ Schedule: {scanner.get_schedule_description()}\n"
            message += f"   👥 Subscribers: {subscribers_count}\n"
            message += f"   Status: {status_emoji} {'Subscribed' if is_subscribed else 'Not subscribed'}\n\n"
            
            inline_keyboard.append([
                InlineKeyboardButton(f"{action_text} to {name}", callback_data=callback_data)
            ])

        keyboard = ReplyKeyboardMarkup(SCANNER_KEYBOARD, resize_keyboard=True)
        inline_markup = InlineKeyboardMarkup(inline_keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        
        # Send inline keyboard as separate message for better UX
        await update.message.reply_text(
            "💡 *Quick Actions:*",
            reply_markup=inline_markup,
            parse_mode='Markdown'
        )

    async def subscribe_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /subscribe command."""
        user_id = update.effective_user.id

        if not context.args:
            await update.message.reply_text("Usage: /subscribe <scanner_name>")
            return

        scanner_name = " ".join(context.args)

        if scanner_name not in self.scanners:
            available = ", ".join(self.scanners.keys())
            await update.message.reply_text(f"Scanner '{scanner_name}' not found. Available: {available}")
            return

        if self.subscribe_user(scanner_name, user_id):
            await update.message.reply_text(f"✅ Subscribed to {scanner_name}")
        else:
            await update.message.reply_text(f"❌ Failed to subscribe to {scanner_name}")

    async def unsubscribe_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /unsubscribe command."""
        user_id = update.effective_user.id

        if not context.args:
            await update.message.reply_text("Usage: /unsubscribe <scanner_name>")
            return

        scanner_name = " ".join(context.args)

        if scanner_name not in self.scanners:
            available = ", ".join(self.scanners.keys())
            await update.message.reply_text(f"Scanner '{scanner_name}' not found. Available: {available}")
            return

        if self.unsubscribe_user(scanner_name, user_id):
            await update.message.reply_text(f"✅ Unsubscribed from {scanner_name}")
        else:
            await update.message.reply_text(f"❌ Failed to unsubscribe from {scanner_name}")

    async def subscriptions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /subscriptions command."""
        user_id = update.effective_user.id
        subs = self.get_user_subscriptions(user_id)

        keyboard = ReplyKeyboardMarkup(MAIN_KEYBOARD, resize_keyboard=True)
        
        if not subs:
            await update.message.reply_text(
                "📋 *Your Subscriptions*\n\n"
                "❌ You are not subscribed to any scanners.\n\n"
                "Use /scanners to browse available scanners and subscribe!",
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
            return

        message = "📋 *Your Active Subscriptions*\n\n"
        inline_keyboard = []

        for scanner_name in subs:
            scanner = self.scanners[scanner_name]
            message += f"� *{scanner_name}*\n"
            message += f"   ⏰ {scanner.get_schedule_description()}\n\n"
            
            inline_keyboard.append([
                InlineKeyboardButton(f"Unsubscribe from {scanner_name}", callback_data=f"unsubscribe_{scanner_name}")
            ])

        inline_markup = InlineKeyboardMarkup(inline_keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        
        await update.message.reply_text(
            "💡 *Quick Actions:*",
            reply_markup=inline_markup,
            parse_mode='Markdown'
        )

    async def run_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /run command."""
        await self.show_run_interface(update, context)

    async def show_run_interface(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show the manual scanner run interface."""
        if not self.scanners:
            keyboard = ReplyKeyboardMarkup(MAIN_KEYBOARD, resize_keyboard=True)
            await update.message.reply_text(
                "❌ No scanners available to run.",
                reply_markup=keyboard
            )
            return

        message = "🔄 *Manual Scanner Execution*\n\n"
        message += "Choose a scanner to run immediately:\n\n"
        
        inline_keyboard = []

        for name, scanner in self.scanners.items():
            subscribers_count = len(self.subscribers.get(name, set()))
            message += f"📈 *{name}*\n"
            message += f"   ⏰ Schedule: {scanner.get_schedule_description()}\n"
            message += f"   👥 Subscribers: {subscribers_count}\n\n"
            
            inline_keyboard.append([
                InlineKeyboardButton(f"🚀 Run {name} Now", callback_data=f"run_{name}")
            ])

        keyboard = ReplyKeyboardMarkup(RUN_SCANNER_KEYBOARD, resize_keyboard=True)
        inline_markup = InlineKeyboardMarkup(inline_keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        
        await update.message.reply_text(
            "⚡ *Run Actions:*",
            reply_markup=inline_markup,
            parse_mode='Markdown'
        )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard button presses."""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        data = query.data
        
        if data.startswith('subscribe_'):
            scanner_name = data[10:]  # Remove 'subscribe_' prefix
            if self.subscribe_user(scanner_name, user_id):
                await query.edit_message_text(
                    f"✅ Successfully subscribed to *{scanner_name}*!\n\n"
                    f"You will receive notifications when this scanner runs.",
                    parse_mode='Markdown'
                )
            else:
                await query.edit_message_text(
                    f"❌ Failed to subscribe to *{scanner_name}*.",
                    parse_mode='Markdown'
                )
                
        elif data.startswith('unsubscribe_'):
            scanner_name = data[12:]  # Remove 'unsubscribe_' prefix
            if self.unsubscribe_user(scanner_name, user_id):
                await query.edit_message_text(
                    f"✅ Successfully unsubscribed from *{scanner_name}*.\n\n"
                    f"You will no longer receive notifications from this scanner.",
                    parse_mode='Markdown'
                )
            else:
                await query.edit_message_text(
                    f"❌ Failed to unsubscribe from *{scanner_name}*.",
                    parse_mode='Markdown'
                )
                
        elif data.startswith('run_'):
            scanner_name = data[4:]  # Remove 'run_' prefix
            await query.edit_message_text(
                f"🚀 Starting manual execution of *{scanner_name}*...\n\n"
                f"Please wait while the scanner runs.",
                parse_mode='Markdown'
            )
            
            # Run the scanner manually
            try:
                results = self.scanners[scanner_name].scan()
                
                if results.get('success', False):
                    embed = results.get('embed', {})
                    title = embed.get('title', f'{scanner_name} Results')
                    fields = embed.get('fields', [])
                    
                    message = f"📊 *{title}* (Manual Run)\n\n"
                    
                    for field in fields:
                        message += f"**{field['name']}**\n"
                        message += f"{field['value']}\n\n"
                    
                    await query.edit_message_text(
                        f"✅ *{scanner_name}* executed successfully!\n\n{message}",
                        parse_mode='Markdown'
                    )
                else:
                    error_msg = results.get('error', 'Unknown error')
                    await query.edit_message_text(
                        f"❌ *{scanner_name}* failed to execute.\n\n"
                        f"Error: {error_msg}",
                        parse_mode='Markdown'
                    )
                    
            except Exception as e:
                await query.edit_message_text(
                    f"❌ Error running *{scanner_name}*.\n\n"
                    f"Exception: {str(e)}",
                    parse_mode='Markdown'
                )
                
        elif data == 'show_scanners':
            await self.scanners_command(update, context)
            
        elif data == 'show_subscriptions':
            await self.subscriptions_command(update, context)

    async def handle_keyboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle keyboard button presses."""
        text = update.message.text
        
        if text == '📊 Scanners':
            await self.scanners_command(update, context)
        elif text == '📋 My Subscriptions':
            await self.subscriptions_command(update, context)
        elif text == '🔄 Run Scanner':
            await self.show_run_interface(update, context)
        elif text == '❓ Help':
            await self.help_command(update, context)
        elif text == '🚪 Close Keyboard':
            keyboard = ReplyKeyboardRemove()
            await update.message.reply_text(
                "⌨️ Keyboard closed. You can always use /start to bring it back!",
                reply_markup=keyboard
            )
        elif text == '📊 List Scanners':
            await self.scanners_command(update, context)
        elif text == '⬅️ Back to Main':
            keyboard = ReplyKeyboardMarkup(MAIN_KEYBOARD, resize_keyboard=True)
            await update.message.reply_text(
                "🏠 *Back to Main Menu*\n\nChoose an option:",
                reply_markup=keyboard,
                parse_mode='Markdown'
            )

    async def run_scanner(self, scanner_name: str):
        """Run a scanner and send results to subscribers."""
        if scanner_name not in self.scanners:
            logger.error(f"Scanner {scanner_name} not found")
            return

        scanner = self.scanners[scanner_name]
        logger.info(f"Running scanner: {scanner_name}")

        try:
            results = scanner.scan()

            if not results.get('success', False):
                error_msg = results.get('error', 'Unknown error')
                logger.error(f"Scanner {scanner_name} failed: {error_msg}")
                return

            # Send results to subscribers
            subscribers = self.subscribers.get(scanner_name, set())
            if not subscribers:
                logger.info(f"No subscribers for {scanner_name}")
                return

            # Format message for Telegram
            embed = results.get('embed', {})
            title = embed.get('title', f'{scanner_name} Results')
            fields = embed.get('fields', [])

            message = f"📊 {title}\n\n"

            for field in fields:
                message += f"**{field['name']}**\n"
                message += f"{field['value']}\n\n"

            # Send to all subscribers
            for user_id in subscribers:
                try:
                    await self.application.bot.send_message(
                        chat_id=user_id,
                        text=message,
                        parse_mode='Markdown'
                    )
                    logger.info(f"Sent results to user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to send message to user {user_id}: {e}")

        except Exception as e:
            logger.error(f"Error running scanner {scanner_name}: {e}")

    def setup_scheduler(self):
        """Set up the scheduler for all scanners."""
        est = pytz.timezone('Europe/Berlin')
        for scanner_name, scanner in self.scanners.items():
            # Parse cron schedule
            try:
                trigger = CronTrigger.from_crontab(scanner.schedule, timezone=est)
                self.scheduler.add_job(
                    self.run_scanner_sync,
                    trigger=trigger,
                    args=[scanner_name],
                    id=f"scanner_{scanner_name}",
                    name=f"Run {scanner_name} scanner"
                )
                logger.info(f"Scheduled {scanner_name} with cron: {scanner.schedule} in EST")
            except Exception as e:
                logger.error(f"Failed to schedule {scanner_name}: {e}")

    def run_scanner_sync(self, scanner_name: str):
        """Run a scanner synchronously (for blocking scheduler)."""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_scanner(scanner_name))
        finally:
            loop.close()

    def run(self):
        """Run the bot."""
        # Create application
        self.application = Application.builder().token(self.token).build()

        # Add command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("scanners", self.scanners_command))
        self.application.add_handler(CommandHandler("subscribe", self.subscribe_command))
        self.application.add_handler(CommandHandler("unsubscribe", self.unsubscribe_command))
        self.application.add_handler(CommandHandler("subscriptions", self.subscriptions_command))
        self.application.add_handler(CommandHandler("run", self.run_command))
        
        # Add callback query handler for inline keyboards
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Add message handler for keyboard buttons
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_keyboard))

        # Set up scheduler
        self.setup_scheduler()
        
        # Start scheduler in a separate thread
        self.scheduler_thread = threading.Thread(target=self.scheduler.start, daemon=True)
        self.scheduler_thread.start()

        logger.info("Bot started successfully")

        # Run the bot (synchronous)
        try:
            self.application.run_polling()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            # Properly shutdown scheduler
            if self.scheduler.running:
                self.scheduler.shutdown(wait=True)
            logger.info("Bot and scheduler shut down gracefully")


def main():
    """Main entry point."""
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
        return

    bot = TradingBot(token)

    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")


if __name__ == '__main__':
    main()