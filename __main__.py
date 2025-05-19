# from utils.selector import StockSelector
import yfinance as yf


if __name__ == "__main__":
    # tickers = ['NVDA', 'MSFT', 'BRK-B', 'PG', 'XOM']
    # stock = StockSelector(tickers)
    df = yf.download("NVDA")
    print(df)
