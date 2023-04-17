import warnings
warnings.filterwarnings("ignore")
import talib
from backtesting import Strategy, Backtest
from backtesting.test import GOOG

class SmaCross(Strategy):
    # Define indicator variables
    n1 = 12
    n2 = 26
    n3 = 9
    
    def init(self):
        # Define indicator variables using talib
        self.macd, self.macd_signal, self.macd_hist = self.I(talib.MACD, self.data.Close, self.n1, self.n2, self.n3)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)

    def next(self):
        # Check buy and sell conditions
        if self.macd[-1] > 20 and self.macd[-1] > self.macd_signal[-1] and self.macd[-2] < self.macd_signal[-2] and self.rsi[-1] > 50:
            self.buy()

        elif self.position and (self.macd[-1] < self.macd_signal[-1] or self.macd[-2] > self.macd_signal[-2] or self.rsi[-1] < 50):
            self.sell()

bt = Backtest(GOOG, SmaCross, cash=10_000, commission=.002)
stats = bt.run()

stats = bt.optimize(n1=range(6, 24, 5), n2=range(13, 52, 5), n3=range(4, 18, 3))
# Print final statistics
print("Final Equity: ", stats["Equity Final [$]"])
print("Peak Equity ", stats["Equity Peak [$]"])
print("Return Rate: ", stats["Return [%]"])
print("Buy and hold return rate: ", stats["Buy & Hold Return [%]"])
print("Sharpe Ratio: ", stats["Sharpe Ratio"])
print("Max Drawdown Rate: ", stats["Max. Drawdown [%]"])
print("Trade Count", stats["# Trades"])