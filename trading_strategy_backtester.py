import pandas as pd
import numpy as np
import yfinance as yf
import ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class TradingStrategyBacktester:
    def __init__(self, symbol='EURUSD=X', interval='15m', period='60d'):
        """
        Initialize the backtester with trading parameters
        """
        self.symbol = symbol
        self.interval = interval
        self.period = period
        self.df = None
        self.trades = []
        self.equity_curve = []
        
    def fetch_data(self):
        """
        Fetch OHLC data from yfinance
        """
        print(f"Fetching {self.period} of {self.interval} data for {self.symbol}...")
        try:
            self.df = yf.download(self.symbol, period=self.period, interval=self.interval, progress=False)
            print(f"Data fetched successfully: {len(self.df)} candles")
            return self.df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def add_indicators(self):
        """
        Add EMA 20 and EMA 50 indicators
        """
        self.df['EMA_20'] = ta.trend.ema_indicator(self.df['Close'], window=20)
        self.df['EMA_50'] = ta.trend.ema_indicator(self.df['Close'], window=50)
        
        # EMA Crossover signals
        self.df['EMA_Cross'] = 0
        self.df.loc[self.df['EMA_20'] > self.df['EMA_50'], 'EMA_Cross'] = 1  # Bullish
        self.df.loc[self.df['EMA_20'] < self.df['EMA_50'], 'EMA_Cross'] = -1  # Bearish
        
        # Detect crossovers
        self.df['EMA_Crossover'] = self.df['EMA_Cross'].diff()
        
        print("Indicators added successfully")
    
    def detect_candlestick_patterns(self):
        """
        Detect Bullish Engulfing, Bearish Engulfing, Morning Star, and Evening Star
        """
        self.df['Pattern'] = 0
        
        for i in range(2, len(self.df)):
            # Bullish Engulfing
            if (self.df['Open'].iloc[i-1] > self.df['Close'].iloc[i-1] and  # Previous candle is bearish
                self.df['Close'].iloc[i] > self.df['Open'].iloc[i-1] and   # Current close > previous open
                self.df['Open'].iloc[i] < self.df['Close'].iloc[i-1]):      # Current open < previous close
                self.df.loc[self.df.index[i], 'Pattern'] = 1
            
            # Bearish Engulfing
            elif (self.df['Open'].iloc[i-1] < self.df['Close'].iloc[i-1] and  # Previous candle is bullish
                  self.df['Close'].iloc[i] < self.df['Open'].iloc[i-1] and     # Current close < previous open
                  self.df['Open'].iloc[i] > self.df['Close'].iloc[i-1]):       # Current open > previous close
                self.df.loc[self.df.index[i], 'Pattern'] = -1
            
            # Morning Star (3-candle pattern)
            elif (i >= 2 and
                  self.df['Close'].iloc[i-2] < self.df['Open'].iloc[i-2] and  # First candle bearish
                  self.df['Close'].iloc[i] > ((self.df['Open'].iloc[i-2] + self.df['Close'].iloc[i-2]) / 2)):
                self.df.loc[self.df.index[i], 'Pattern'] = 1
            
            # Evening Star (3-candle pattern)
            elif (i >= 2 and
                  self.df['Close'].iloc[i-2] > self.df['Open'].iloc[i-2] and  # First candle bullish
                  self.df['Close'].iloc[i] < ((self.df['Open'].iloc[i-2] + self.df['Close'].iloc[i-2]) / 2)):
                self.df.loc[self.df.index[i], 'Pattern'] = -1
        
        print("Candlestick patterns detected")
    
    def identify_entry_signals(self):
        """
        Identify entry signals: EMA crossover + Candlestick pattern
        """
        self.df['Entry_Signal'] = 0
        
        for i in range(1, len(self.df)):
            # Bullish signal: EMA 20 crosses above EMA 50 + Bullish pattern
            if (self.df['EMA_Crossover'].iloc[i] > 0 and self.df['Pattern'].iloc[i] == 1):
                self.df.loc[self.df.index[i], 'Entry_Signal'] = 1
            
            # Bearish signal: EMA 20 crosses below EMA 50 + Bearish pattern
            elif (self.df['EMA_Crossover'].iloc[i] < 0 and self.df['Pattern'].iloc[i] == -1):
                self.df.loc[self.df.index[i], 'Entry_Signal'] = -1
        
        print("Entry signals identified")
    
    def identify_market_structure(self):
        """
        Identify market structure on the line chart
        HH, HL = Bullish | LH, LL = Bearish
        """
        self.df['Market_Structure'] = 'NONE'
        
        for i in range(2, len(self.df)):
            # Bullish Structure (HH, HL)
            if (self.df['High'].iloc[i] > self.df['High'].iloc[i-1] and
                self.df['Low'].iloc[i] > self.df['Low'].iloc[i-1]):
                self.df.loc[self.df.index[i], 'Market_Structure'] = 'BULLISH'
            
            # Bearish Structure (LH, LL)
            elif (self.df['High'].iloc[i] < self.df['High'].iloc[i-1] and
                  self.df['Low'].iloc[i] < self.df['Low'].iloc[i-1]):
                self.df.loc[self.df.index[i], 'Market_Structure'] = 'BEARISH'
        
        print("Market structure identified")
    
    def identify_liquidity_sweep(self):
        """
        Identify liquidity sweeps (price touching previous swing high/low)
        """
        self.df['Liquidity_Sweep'] = 0
        
        for i in range(5, len(self.df)):
            # Find previous swing high and low
            prev_high = self.df['High'].iloc[max(0, i-5):i].max()
            prev_low = self.df['Low'].iloc[max(0, i-5):i].min()
            
            # Liquidity sweep if price touches and reverses
            if (self.df['Low'].iloc[i] <= prev_low and self.df['Close'].iloc[i] > prev_low):
                self.df.loc[self.df.index[i], 'Liquidity_Sweep'] = -1  # Bullish reversal
            elif (self.df['High'].iloc[i] >= prev_high and self.df['Close'].iloc[i] < prev_high):
                self.df.loc[self.df.index[i], 'Liquidity_Sweep'] = 1  # Bearish reversal
        
        print("Liquidity sweeps identified")
    
    def backtest(self, risk_reward_ratio=2, initial_capital=10000, risk_per_trade=0.02):
        """
        Execute backtest with position sizing based on risk/reward
        """
        print(f"\n{'='*60}")
        print(f"BACKTESTING PARAMETERS:")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Timeframe: {self.interval}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Risk Per Trade: {risk_per_trade*100}%")
        print(f"Risk/Reward Ratio: 1:{risk_reward_ratio}")
        print(f"{'='*60}\n")
        
        balance = initial_capital
        position = None
        entry_price = 0
        entry_index = 0
        self.equity_curve = [initial_capital]
        
        for i in range(1, len(self.df)):
            current_price = self.df['Close'].iloc[i]
            current_time = self.df.index[i]
            
            # Check for entry signals
            if position is None and self.df['Entry_Signal'].iloc[i] != 0:
                # Check liquidity sweep before entry
                if self.df['Liquidity_Sweep'].iloc[i] == 0:
                    position = 'LONG' if self.df['Entry_Signal'].iloc[i] == 1 else 'SHORT'
                    entry_price = current_price
                    entry_index = i
                    
                    # Find swing high/low for target
                    if position == 'LONG':
                        swing_high = self.df['High'].iloc[max(0, i-10):i].max()
                        stop_loss = entry_price * 0.99  # 1% stop loss
                        take_profit = entry_price + (swing_high - entry_price) * risk_reward_ratio
                    else:  # SHORT
                        swing_low = self.df['Low'].iloc[max(0, i-10):i].min()
                        stop_loss = entry_price * 1.01  # 1% stop loss
                        take_profit = entry_price - (entry_price - swing_low) * risk_reward_ratio
            
            # Check for exit signals
            elif position is not None:
                if position == 'LONG':
                    if current_price >= take_profit:
                        # Take Profit
                        profit = (current_price - entry_price) * 100 / entry_price
                        balance += balance * (profit / 100)
                        self.trades.append({
                            'Entry Time': self.df.index[entry_index],
                            'Exit Time': current_time,
                            'Type': 'LONG',
                            'Entry Price': entry_price,
                            'Exit Price': current_price,
                            'Profit %': profit,
                            'Result': 'WIN'
                        })
                        position = None
                    
                    elif current_price <= stop_loss:
                        # Stop Loss
                        loss = (current_price - entry_price) * 100 / entry_price
                        balance += balance * (loss / 100)
                        self.trades.append({
                            'Entry Time': self.df.index[entry_index],
                            'Exit Time': current_time,
                            'Type': 'LONG',
                            'Entry Price': entry_price,
                            'Exit Price': current_price,
                            'Profit %': loss,
                            'Result': 'LOSS'
                        })
                        position = None
                
                elif position == 'SHORT':
                    if current_price <= take_profit:
                        # Take Profit
                        profit = (entry_price - current_price) * 100 / entry_price
                        balance += balance * (profit / 100)
                        self.trades.append({
                            'Entry Time': self.df.index[entry_index],
                            'Exit Time': current_time,
                            'Type': 'SHORT',
                            'Entry Price': entry_price,
                            'Exit Price': current_price,
                            'Profit %': profit,
                            'Result': 'WIN'
                        })
                        position = None
                    
                    elif current_price >= stop_loss:
                        # Stop Loss
                        loss = (current_price - entry_price) * 100 / entry_price
                        balance += balance * (loss / 100)
                        self.trades.append({
                            'Entry Time': self.df.index[entry_index],
                            'Exit Time': current_time,
                            'Type': 'SHORT',
                            'Entry Price': entry_price,
                            'Exit Price': current_price,
                            'Profit %': loss,
                            'Result': 'LOSS'
                        })
                        position = None
            
            self.equity_curve.append(balance)
        
        self.print_backtest_results(initial_capital, balance)
    
    def print_backtest_results(self, initial_capital, final_balance):
        """
        Print backtest results and statistics
        """
        if not self.trades:
            print("No trades executed during backtest period")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        total_return = ((final_balance - initial_capital) / initial_capital) * 100
        winning_trades = len(trades_df[trades_df['Result'] == 'WIN'])
        losing_trades = len(trades_df[trades_df['Result'] == 'LOSS'])
        win_rate = (winning_trades / len(trades_df) * 100) if len(trades_df) > 0 else 0
        
        avg_win = trades_df[trades_df['Result'] == 'WIN']['Profit %'].mean()
        avg_loss = abs(trades_df[trades_df['Result'] == 'LOSS']['Profit %'].mean())
        
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS:")
        print(f"{'='*60}")
        print(f"Initial Capital:     ${initial_capital:,.2f}")
        print(f"Final Balance:       ${final_balance:,.2f}")
        print(f"Total Return:        {total_return:.2f}%")
        print(f"Total Trades:        {len(trades_df)}")
        print(f"Winning Trades:      {winning_trades} ({win_rate:.2f}%)")
        print(f"Losing Trades:       {losing_trades}")
        print(f"Average Win:         {avg_win:.2f}%")
        print(f"Average Loss:        {avg_loss:.2f}%")
        print(f"Profit Factor:       {(winning_trades / max(1, losing_trades)):.2f}")
        print(f"{'='*60}\n")
        
        print("TRADES LOG:")
        print(trades_df.to_string(index=False))
    
    def plot_results(self):
        """
        Plot the backtest results with EMA and price action
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot 1: Price with EMAs
        ax1.plot(self.df.index, self.df['Close'], label='Close Price', linewidth=1.5, color='black')
        ax1.plot(self.df.index, self.df['EMA_20'], label='EMA 20', linewidth=1, color='blue', alpha=0.7)
        ax1.plot(self.df.index, self.df['EMA_50'], label='EMA 50', linewidth=1, color='red', alpha=0.7)
        
        # Plot entry signals
        bullish_signals = self.df[self.df['Entry_Signal'] == 1]
        bearish_signals = self.df[self.df['Entry_Signal'] == -1]
        
        ax1.scatter(bullish_signals.index, bullish_signals['Close'], marker='^', color='green', s=100, label='Bullish Signal')
        ax1.scatter(bearish_signals.index, bearish_signals['Close'], marker='v', color='red', s=100, label='Bearish Signal')
        
        ax1.set_title('Trading Strategy - Price Action with EMA Crossover', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=10)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity Curve
        ax2.plot(range(len(self.equity_curve)), self.equity_curve, linewidth=2, color='green')
        ax2.fill_between(range(len(self.equity_curve)), self.equity_curve, alpha=0.3, color='green')
        ax2.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Candles', fontsize=10)
        ax2.set_ylabel('Account Balance ($)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('trading_backtest_results.png', dpi=300, bbox_inches='tight')
        print("Chart saved as 'trading_backtest_results.png'")
        plt.show()

# Run the Backtest
if __name__ == "__main__":
    # Initialize backtester
    backtester = TradingStrategyBacktester(symbol='EURUSD=X', interval='15m', period='60d')
    
    # Fetch data
    backtester.fetch_data()
    
    if backtester.df is not None:
        # Add all indicators and patterns
        backtester.add_indicators()
        backtester.identify_market_structure()
        backtester.detect_candlestick_patterns()
        backtester.identify_liquidity_sweep()
        backtester.identify_entry_signals()
        
        # Run backtest with 1:2 risk/reward ratio
        backtester.backtest(risk_reward_ratio=2, initial_capital=10000, risk_per_trade=0.02)
        
        # Plot results
        backtester.plot_results()