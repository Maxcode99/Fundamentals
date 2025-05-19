import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import os
import xlrd


class StockSelector():

    def __init__(self, list_of_stocks_route):

        self.stock_route = list_of_stocks_route
        self.dataFrame = pd.read_csv(self.stock_route)
        self.stock_list = self.dataFrame["Ticker"].tolist()
        test_ticket = self.stock_list[0]
        balance, estado = self.get_historical_financial(test_ticket)
        self.stock_dates = ['2024-12-31', '2023-12-31', '2022-12-31', '2021-12-31', '2020-12-31']



    def get_historical_price(self, ticker: str, start: str):
        end = start + datetime.timedelta(days=4)
        price = pd.DataFrame(yf.Ticker(ticker).history(start=start, end=end)["Close"])
        return price.iloc[0, 0]

    def get_historical_financial(self, ticker: str):
        stock = yf.Ticker(ticker)
        balance = stock.balance_sheet
        estado = stock.income_stmt

        return balance, estado

    def getting_eps(self):

        conjunto_acciones = {}

        for stock in self.stock_list:
            print("The stocks I used are: " + str(stock))

            try:
                t = stock
                balance, estado = self.get_historical_financial(t)

                income = estado.loc["Net Income"].values
                n_shares = balance.loc["Ordinary Shares Number"].values
                eps = income / n_shares
                dict_eps = {i: j for i, j in zip(estado.columns, eps)}

                stock_data = {} 
                for value in range(len(self.stock_dates)):
                    key = self.stock_dates[value] 
                    stock_data[key] = list(dict_eps.values())[value] 

                conjunto_acciones[stock] = stock_data 

            except Exception as e:
                print("Error processing stock:", stock, e)
                continue


        df_acciones = pd.DataFrame.from_dict(conjunto_acciones, orient='index')
        df_acciones.to_csv("assets/eps_data.csv")




    def getting_solvency(self):

        conjunto_acciones = {}

        for stock in self.stock_list:
            print("The stocks I used are: " + str(stock))

            try:
                t = stock
                balance, estado = self.get_historical_financial(t)

                assets = balance.loc['Total Assets']
                liabilities = balance.loc['Current Liabilities']
                solvency = assets / liabilities

                stock_data = {}  
                for value in range(len(self.stock_dates)):
                    key = self.stock_dates[value]  
                    stock_data[key] = solvency.values[value]  

                conjunto_acciones[stock] = stock_data 

            except Exception as e:
                print("Error processing stock:", stock, e)
                continue

        df_acciones = pd.DataFrame.from_dict(conjunto_acciones, orient='index')
        df_acciones.to_csv("assets/solvency_data.csv")



    def getting_roe(self):



        conjunto_acciones = {}

        for stock in self.stock_list:
            print("The stocks I used are: " + str(stock))
    
            try:
                t = stock
                balance, estado = self.get_historical_financial(t)

                #ROE
                income = estado.loc["Net Income"].values
                equity = balance.loc['Stockholders Equity']
                roe = income/equity*100


                stock_data = {} 
                for value in range(len(self.stock_dates)):
                    key = self.stock_dates[value] 
                    stock_data[key] = roe.values[value] 

                conjunto_acciones[stock] = stock_data  

            except Exception as e:
                print("Error processing stock:", stock, e)
                continue

        df_acciones = pd.DataFrame.from_dict(conjunto_acciones, orient='index')
        df_acciones.to_csv("assets/roe_data.csv")


    def getting_pbv(self):


        conjunto_acciones = {}

        for stock in self.stock_list:
            print("The stocks I used are: " + str(stock))
            # new_breaker += 1

            try:
                t = stock
                balance, estado = self.get_historical_financial(t)

                #PBV
                price = [self.get_historical_price(t, date) for date in balance.columns]
                n_shares = balance.loc["Ordinary Shares Number"].values
                market_cap = n_shares * price
                book_value = balance.loc['Tangible Book Value'].values
                dict_pbv = {i : j for i, j in zip(estado.columns, market_cap/book_value)}


                stock_data = {}  # Diccionario para los datos de este stock
                for value in range(len(self.stock_dates)):
                    key = self.stock_dates[value]  # Fecha como clave
                    stock_data[key] = list(dict_pbv.values())[value]  # Asignación de valor de solvencia

                conjunto_acciones[stock] = stock_data  # Asignación al stock correspondiente

            except Exception as e:
                print("Error processing stock:", stock, e)
                continue

        df_acciones = pd.DataFrame.from_dict(conjunto_acciones, orient='index')
        df_acciones.to_csv("assets/pbv_data.csv")






if __name__ == "__main__":

    route = "../assets/stock_info.csv"
    stockSelector = StockSelector(route)
    print(stockSelector)
    print(stockSelector.stock_route)
    print(stockSelector.dataFrame)
    print(stockSelector.stock_list)
    stockSelector.getting_eps()