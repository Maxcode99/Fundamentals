import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import time


class StockSelector():

    def __init__(self, list_of_stocks: list, start: str = None, end: str = None, delay: float = 1.5):

        all_data = []

        for ticker in list_of_stocks:
            try:
                print(f"Downloading: {ticker}")
                df = yf.download(ticker, start=start, end=end, progress=False)
                df["Ticker"] = ticker
                all_data.append(df)
                time.sleep(delay)  # wait between requests to avoid rate limit
            except Exception as e:
                print(f"Failed to download {ticker}: {e}")

        if not all_data:
            raise ValueError("No data was downloaded. Check tickers or rate limits.")

        # Combine all into a single DataFrame
        self.dataFrame = pd.concat(all_data, keys=list_of_stocks, names=["Ticker", "Date"])
        self.dataFrame = self.dataFrame.reset_index().set_index(["Date", "Ticker"]).sort_index()

        # Pivot to get Close and Volume across tickers
        self.prices = self.dataFrame["Close"].unstack()
        self.volumes = self.dataFrame["Volume"].unstack()
        self.returns = self.prices.pct_change()
        self.consolidated_data = pd.concat(
            [self.prices, self.volumes, self.returns],
            axis=1,
            keys=["Prices", "Volumes", "Returns"]
        )


    def get_historical_financial(self, ticker: str):
        ticker_obj = yf.Ticker(ticker)
        balance = ticker_obj.balance_sheet
        estado = ticker_obj.financials
        return balance, estado, ticker_obj

    def get_historical_price(self, ticker: str, start: str):
        start_date = pd.to_datetime(start)  # Asegurándose de que 'start' sea un objeto datetime
        end = start_date + datetime.timedelta(days=4)
        price = pd.DataFrame(
            yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))["Close"])
        return price.iloc[0, 0]

    def calculate_financials(self, ticker: str):
        balance, estado, ticker_obj = self.get_historical_financial(ticker)

        # Asumimos que la última columna es la más reciente
        latest_date = balance.columns[0]

        # Corrección para asegurarse de que latest_date es un string en formato adecuado
        latest_date_str = pd.to_datetime(latest_date).strftime('%Y-%m-%d')

        # EPS
        income = estado.loc['Net Income', latest_date]
        n_shares = balance.loc['Ordinary Shares Number', latest_date]
        eps = income / n_shares

        # PER
        price = self.get_historical_price(ticker, latest_date_str)
        per = price / eps

        # MARKET CAP
        market_cap = n_shares * price / 1_000_000

        # PBV
        book_value = balance.loc['Tangible Book Value', latest_date]
        pbv = (market_cap * 1000000) / book_value

        # SOLVENCIA
        assets = balance.loc['Total Assets', latest_date]

        if ticker == 'BRK-B':
            items_to_sum = [
                'Current Debt And Capital Lease Obligation',
                'Current Debt',
                'Other Current Borrowings',
                'Payables And Accrued Expenses'
            ]

            liabilities = balance.loc[items_to_sum, latest_date].sum()
        else:
            liabilities = balance.loc['Current Liabilities', latest_date]

        solvency = assets / liabilities

        # ROE
        equity = balance.loc['Stockholders Equity', latest_date]
        roe = (income / equity) * 100

        # Beta
        beta = ticker_obj.info.get('beta', 'N/A')  # Usamos .get para manejar posibles None

        # Corrección en la creación del DataFrame para incluir 'Beta' correctamente
        financials_df = pd.DataFrame({
            "Fundamental": ["EPS", "PER", "Market Cap", "PBV", "Solvency", "ROE", "Beta"],
            "Value": [f"{eps:.4f}", f"{per:.4f}", f"{market_cap:.2f}", f"{pbv:.6f}", f"{solvency:.4f}", f"{roe:.4f}",
                      f"{beta if beta != 'N/A' else beta:.2f}" if beta != 'N/A' else "N/A"]
        })

        return financials_df

    # Función Sharpe
    def maximize_sharpe(rends: pd.DataFrame, rf: float, n_sims: int):

        expected_return = rends.mean()
        cov = rends.cov()
        n_stocks = len(rends.columns)
        w, sharpe = [], []
        for _ in range(n_sims):
            # Inicializar pesos
            temp_w = np.random.uniform(0, 1, n_stocks)
            temp_w = temp_w / np.sum(temp_w)

            # Calcula rend y vol
            ret = np.dot(temp_w, expected_return)
            vol = np.sqrt(temp_w.reshape(1, -1) @ cov @ temp_w)

            # Calcular ratio de sharpe
            temp_sharpe = (ret - rf) / vol

            # Guardar info
            w.append(temp_w)
            sharpe.append(temp_sharpe)

        return w[np.argmax(sharpe)], np.max(sharpe)

    ### Función min semivar pesos

    def min_semivar(rends: pd.DataFrame, rf: float, n_sims: int, downside_risks, correlation_matrix):

        expected_return = rends.mean()
        cov = rends.cov()
        n_stocks = len(rends.columns)
        w, semi = [], []
        for _ in range(n_sims):
            # Inicializar pesos
            temp_w = np.random.uniform(0, 1, n_stocks)
            temp_w = temp_w / np.sum(temp_w)

            # Calcula rend y vol
            ret = np.dot(temp_w, expected_return)
            vol = np.sqrt(temp_w.reshape(1, -1) @ cov @ temp_w)

            # Calcular semivar
            downsiderisk = pd.DataFrame.from_dict(downside_risks, orient='index').T
            transpose_d = downsiderisk.T
            downside_matrix = pd.DataFrame(np.dot(transpose_d, downsiderisk))
            semi_matrix = np.multiply(downside_matrix, correlation_matrix)

            ### PORTSEMI
            y = temp_w.T
            x = np.dot(semi_matrix, temp_w)
            port_semi_shit = np.dot(y, x)

            # Guardar info
            w.append(temp_w)
            semi.append(port_semi_shit)

        return w[np.argmin(semi)], np.min(semi)

    ### Función Omega Max
    def maximize_omega(rends: pd.DataFrame, rf: float, n_sims: int, omega_ratios):

        expected_return = rends.mean()
        cov = rends.cov()
        n_stocks = len(rends.columns)
        w, omega = [], []
        for _ in range(n_sims):
            # Inicializar pesos
            temp_w = np.random.uniform(0, 1, n_stocks)
            temp_w = temp_w / np.sum(temp_w)

            # Calcula rend y vol
            ret = np.dot(temp_w, expected_return)
            vol = np.sqrt(temp_w.reshape(1, -1) @ cov @ temp_w)

            # Calcular ratio de omega

            romega = pd.DataFrame.from_dict(omega_ratios, orient='index')
            port_omega_shit = (temp_w * romega.values.flatten()).sum()

            # Guardar info
            w.append(temp_w)
            omega.append(port_omega_shit)

        return w[np.argmax(omega)], np.max(omega)

    ### Función min VaR
    # 7. Portafolio de mínima varianza
    def min_VaR(rends: pd.DataFrame, n_sims: int):

        n_stocks = len(rends.columns)
        w, value_at_risk = [], []

        for _ in range(n_sims):
            # Inicializar pesos
            temp_w = np.random.uniform(0, 1, n_stocks)
            temp_w = temp_w / np.sum(temp_w)

            # Calcula VaR
            rend_port = (temp_w * rends.values).sum(axis=1)
            temp_var = np.percentile(rend_port, 100 * (1 - .95))

            # Guardar info
            w.append(temp_w)
            value_at_risk.append(temp_var)

        return w[np.argmin(value_at_risk)], np.min(value_at_risk)

    ### función maximizar sortino ratio
    def maximize_sortino_ratio(rends: pd.DataFrame, rf: float, n_sims: int):
        expected_return = rends.mean()
        n_stocks = len(rends.columns)
        w, sortino_ratios = [], []
        for _ in range(n_sims):
            # Inicializar pesos
            temp_w = np.random.uniform(0, 1, n_stocks)
            temp_w = temp_w / np.sum(temp_w)

            # Calcular rendimiento del portafolio
            ret = np.dot(temp_w, expected_return)

            # Calcular la desviación estándar de los rendimientos negativos
            negative_rends = rends @ temp_w
            negative_rends = negative_rends[negative_rends < 0]
            downside_vol = np.std(negative_rends)

            # Calcular ratio de Sortino
            temp_sortino = (ret - rf) / downside_vol if downside_vol > 0 else np.nan

            # Guardar info
            w.append(temp_w)
            sortino_ratios.append(temp_sortino)

        max_index = np.nanargmax(sortino_ratios)  # Encuentra el índice del máximo ratio de Sortino, ignorando NaNs
        return w[max_index], np.nanmax(sortino_ratios)  # Devuelve los pesos óptimos y el máximo ratio de Sortino+

    def VaR_portfolio(self, position, conf, long):

        if long == True:
            VaR = np.percentile(self.returns, (1 - conf) * 100)
            Es_if = self.returns < VaR
            ES = self.returns[Es_if].mean()

        else:
            VaR = np.percentile(self.returns, (conf) * 100)
            Es_if = self.returns > VaR
            ES = self.returns[Es_if].mean()

        VaR_cash = VaR * position
        Es_cash = ES * position

        metricas_var = pd.DataFrame({
            'VaR (%)': [VaR * 100],
            'VaR ($)': VaR_cash,
            'ES (%)': ES * 100,
            'ES ($)': Es_cash
        })
        return metricas_var

if __name__ == "__main__":
    tickers = ['NVDA', 'MSFT', 'PG', 'XOM']
    stock = StockSelector(tickers)
    print(stock.dataFrame)

