import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

class DataHandler:
    import yfinance as yf
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    def __init__(self, symbol, start, end):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = self.download_data()
        self.returns = self.calculate_log_returns()

    def download_data(self):
        """Download adjusted close prices from Yahoo Finance."""
        data = yf.download(self.symbol, start=self.start, end=self.end, progress=False)
        return data["Adj Close"]

    def calculate_log_returns(self):
        """Calculate log returns of the adjusted close prices."""
        return np.log(self.data / self.data.shift(1)).dropna()
