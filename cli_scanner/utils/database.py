"""
Database utility for storing earnings scanner results in SQLite.
"""

import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import os

logger = logging.getLogger(__name__)


class ScannerDatabase:
    """Handle SQLite database operations for scanner results."""
    
    def __init__(self, db_path: str = "scanner_results.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table for scan runs (metadata about each scan execution)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scan_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_date TEXT,
                    scan_type TEXT NOT NULL,
                    total_recommended INTEGER DEFAULT 0,
                    total_near_misses INTEGER DEFAULT 0
                )
            ''')
            
            # Table for stock analysis results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    tier INTEGER,
                    status TEXT NOT NULL,
                    reason TEXT,
                    price REAL,
                    volume REAL,
                    win_rate REAL,
                    win_quarters INTEGER,
                    iv_rv_ratio REAL,
                    term_structure REAL,
                    spy_iv_rv REAL,
                    iv_rv_pass_threshold REAL,
                    iv_rv_near_miss_threshold REAL,
                    sigma_baseline_1y REAL,
                    sigma_short_leg REAL,
                    sigma_short_leg_fair REAL,
                    actual_to_fair_ratio REAL,
                    expected_move_dollars REAL,
                    expected_move_pct REAL,
                    open_interest INTEGER,
                    days_to_expiry INTEGER,
                    earnings_date TEXT,
                    timing TEXT,
                    FOREIGN KEY (run_id) REFERENCES scan_runs(run_id)
                )
            ''')
            
            # Table for iron fly strategies (optional)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS iron_fly_strategies (
                    strategy_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id INTEGER NOT NULL,
                    expiration TEXT,
                    short_put_strike REAL,
                    short_call_strike REAL,
                    long_put_strike REAL,
                    long_call_strike REAL,
                    total_credit REAL,
                    total_debit REAL,
                    net_credit REAL,
                    lower_breakeven REAL,
                    upper_breakeven REAL,
                    risk_reward_ratio REAL,
                    FOREIGN KEY (result_id) REFERENCES stock_results(result_id)
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker ON stock_results(ticker)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_run_id ON stock_results(run_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON scan_runs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON stock_results(status)')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_scan_results(self, recommended: List[str], near_misses: List[Tuple[str, str]], 
                         stock_metrics: Dict[str, Dict], input_date: Optional[str] = None,
                         iron_fly_data: Optional[Dict[str, Dict]] = None) -> int:
        """
        Save scan results to the database.
        
        Args:
            recommended: List of recommended ticker symbols
            near_misses: List of tuples (ticker, reason) for near misses
            stock_metrics: Dictionary mapping ticker to metrics dictionary
            input_date: Optional input date string (MM/DD/YYYY format)
            iron_fly_data: Optional dictionary mapping ticker to iron fly strategy data
        
        Returns:
            run_id: The ID of the created scan run
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert scan run record
            timestamp = datetime.now(timezone.utc).isoformat()
            cursor.execute('''
                INSERT INTO scan_runs (timestamp, input_date, scan_type, total_recommended, total_near_misses)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, input_date, 'scan', len(recommended), len(near_misses)))
            
            run_id = cursor.lastrowid
            
            # Save ALL stocks in stock_metrics, not just recommendations and near misses
            recommended_set = set(recommended)
            near_miss_set = {ticker for ticker, _ in near_misses}
            
            # Insert stock results for all stocks
            for ticker, metrics in stock_metrics.items():
                # Determine status
                if ticker in recommended_set:
                    status = 'recommended'
                    tier = metrics.get('tier', 1)
                elif ticker in near_miss_set:
                    status = 'near_miss'
                    tier = 0
                else:
                    # Stock that failed all checks
                    status = 'fail'
                    tier = 0
                
                # Get reason
                reason = None
                if ticker in recommended_set:
                    reason = metrics.get('reason', 'Tier 1 Trade' if tier == 1 else 'Tier 2 Trade')
                elif ticker in near_miss_set:
                    for tick, reason_text in near_misses:
                        if tick == ticker:
                            reason = reason_text
                            break
                else:
                    # Get reason from metrics if available, otherwise use default
                    reason = metrics.get('reason', 'Failed validation')
                
                # Extract earnings_date and timing if available in metrics
                earnings_date = None
                timing = None
                
                # Insert stock result
                cursor.execute('''
                    INSERT INTO stock_results (
                        run_id, ticker, tier, status, reason, price, volume,
                        win_rate, win_quarters, iv_rv_ratio, term_structure,
                        spy_iv_rv, iv_rv_pass_threshold, iv_rv_near_miss_threshold,
                        sigma_baseline_1y, sigma_short_leg, sigma_short_leg_fair,
                        actual_to_fair_ratio, expected_move_dollars, expected_move_pct,
                        open_interest, days_to_expiry, earnings_date, timing
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id, ticker, tier, status, reason,
                    metrics.get('price'),
                    metrics.get('volume'),
                    metrics.get('win_rate'),
                    metrics.get('win_quarters'),
                    metrics.get('iv_rv_ratio'),
                    metrics.get('term_structure'),
                    metrics.get('spy_iv_rv'),
                    metrics.get('iv_rv_pass_threshold'),
                    metrics.get('iv_rv_near_miss_threshold'),
                    metrics.get('sigma_baseline_1y'),
                    metrics.get('sigma_short_leg'),
                    metrics.get('sigma_short_leg_fair'),
                    metrics.get('actual_to_fair_ratio'),
                    metrics.get('expected_move_dollars'),
                    metrics.get('expected_move_pct'),
                    metrics.get('open_interest'),
                    metrics.get('days_to_expiry'),
                    earnings_date,
                    timing
                ))
                
                result_id = cursor.lastrowid
                
                # Insert iron fly data if available
                if iron_fly_data and ticker in iron_fly_data:
                    fly = iron_fly_data[ticker]
                    if 'error' not in fly:
                        cursor.execute('''
                            INSERT INTO iron_fly_strategies (
                                result_id, expiration, short_put_strike, short_call_strike,
                                long_put_strike, long_call_strike, total_credit, total_debit,
                                net_credit, lower_breakeven, upper_breakeven, risk_reward_ratio
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            result_id,
                            fly.get('expiration'),
                            fly.get('short_put_strike'),
                            fly.get('short_call_strike'),
                            fly.get('long_put_strike'),
                            fly.get('long_call_strike'),
                            fly.get('total_credit'),
                            fly.get('total_debit'),
                            fly.get('net_credit'),
                            fly.get('lower_breakeven'),
                            fly.get('upper_breakeven'),
                            fly.get('risk_reward_ratio')
                        ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved scan results to database (run_id: {run_id}, "
                       f"{len(recommended)} recommended, {len(near_misses)} near misses)")
            return run_id
            
        except Exception as e:
            logger.error(f"Error saving scan results to database: {e}")
            if conn:
                conn.rollback()
                conn.close()
            raise
    
    def save_analyze_result(self, ticker: str, metrics: Dict, 
                          iron_fly_data: Optional[Dict] = None) -> int:
        """
        Save analyze mode result to the database.
        
        Args:
            ticker: Ticker symbol analyzed
            metrics: Metrics dictionary from analyze_ticker
            iron_fly_data: Optional iron fly strategy data
        
        Returns:
            run_id: The ID of the created scan run
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert scan run record
            timestamp = datetime.now(timezone.utc).isoformat()
            cursor.execute('''
                INSERT INTO scan_runs (timestamp, input_date, scan_type, total_recommended, total_near_misses)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, None, 'analyze', 0, 0))
            
            run_id = cursor.lastrowid
            
            # Determine status
            if metrics.get('pass', False):
                status = 'recommended'
                tier = metrics.get('tier', 1)
            elif metrics.get('near_miss', False):
                status = 'near_miss'
                tier = 0
            else:
                status = 'fail'
                tier = 0
            
            # Insert stock result
            cursor.execute('''
                INSERT INTO stock_results (
                    run_id, ticker, tier, status, reason, price, volume,
                    win_rate, win_quarters, iv_rv_ratio, term_structure,
                    spy_iv_rv, iv_rv_pass_threshold, iv_rv_near_miss_threshold,
                    sigma_baseline_1y, sigma_short_leg, sigma_short_leg_fair,
                    actual_to_fair_ratio, expected_move_dollars, expected_move_pct,
                    open_interest, days_to_expiry, earnings_date, timing
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id, ticker, tier, status, metrics.get('reason'),
                metrics.get('price'),
                metrics.get('volume'),
                metrics.get('win_rate'),
                metrics.get('win_quarters'),
                metrics.get('iv_rv_ratio'),
                metrics.get('term_structure'),
                metrics.get('spy_iv_rv'),
                metrics.get('iv_rv_pass_threshold'),
                metrics.get('iv_rv_near_miss_threshold'),
                metrics.get('sigma_baseline_1y'),
                metrics.get('sigma_short_leg'),
                metrics.get('sigma_short_leg_fair'),
                metrics.get('actual_to_fair_ratio'),
                metrics.get('expected_move_dollars'),
                metrics.get('expected_move_pct'),
                metrics.get('open_interest'),
                metrics.get('days_to_expiry'),
                None,
                'Manual Check'
            ))
            
            result_id = cursor.lastrowid
            
            # Insert iron fly data if available
            if iron_fly_data and 'error' not in iron_fly_data:
                cursor.execute('''
                    INSERT INTO iron_fly_strategies (
                        result_id, expiration, short_put_strike, short_call_strike,
                        long_put_strike, long_call_strike, total_credit, total_debit,
                        net_credit, lower_breakeven, upper_breakeven, risk_reward_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_id,
                    iron_fly_data.get('expiration'),
                    iron_fly_data.get('short_put_strike'),
                    iron_fly_data.get('short_call_strike'),
                    iron_fly_data.get('long_put_strike'),
                    iron_fly_data.get('long_call_strike'),
                    iron_fly_data.get('total_credit'),
                    iron_fly_data.get('total_debit'),
                    iron_fly_data.get('net_credit'),
                    iron_fly_data.get('lower_breakeven'),
                    iron_fly_data.get('upper_breakeven'),
                    iron_fly_data.get('risk_reward_ratio')
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved analyze result for {ticker} to database (run_id: {run_id})")
            return run_id
            
        except Exception as e:
            logger.error(f"Error saving analyze result to database: {e}")
            if conn:
                conn.rollback()
                conn.close()
            raise

