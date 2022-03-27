from cgi import test
import imp
from nsepy import get_history
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from datetime import datetime as dt
# from Model import Model
from sklearn.metrics import mean_squared_error
# from RNNModel import RNNModel
# from GRU_model import get_model
# from BILSTM_model import get_model
# from LSTMModel import get_model
from Hybrid_GRU_LSTM import get_model

# from CNNModel import get_model

class Stock:

    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.stk_data = get_history(symbol=self.symbol, start=self.start_date, end=self.end_date)
        
        self.stk_data['Date'] = self.stk_data.index
        self.data2 = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
        self.data2['Date'] = self.stk_data['Date']
        self.data2['Open'] = self.stk_data['Open']
        self.data2['High'] = self.stk_data['High']
        self.data2['Low'] = self.stk_data['Low']
        self.data2['Close'] = self.stk_data['Close']
        self.scaler = None
        self.predeicted_stock_price = None
        self.X_train, self.y_train = self.preprocess()
        print(self.data2)
        print(self.data2)

    def get_time_series(self):
        return self.data2[['Date', 'Open']].copy()

    def preprocess(self):
        train_set = self.data2.iloc[:, 1:2].values
        # print(self.data2)
        # print(train_set)

        # self.scaler = MinMaxScaler(feature_range=(1,2))
        self.scaler = StandardScaler()

        train_set_scaled = self.scaler.fit_transform(train_set)
        X_train = []
        y_train = []
        for i in range(60, len(train_set_scaled)):
            X_train.append(train_set_scaled[i-60:i, 0])
            y_train.append(train_set_scaled[i, 0])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))
        print(X_train.shape)
        print(X_train)
        print(y_train.shape)
        print(y_train)

        return X_train, y_train



    def predict(self):

        # testdataframe= get_history(symbol='HDFCBANK',start=dt.datetime(2021,1,1),end=dt.datetime(2022,1,1))
        # testdataframe['Date'] = testdataframe.index
        # testdata = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
        # testdata['Date'] = testdataframe['Date']
        # testdata['Open'] = testdataframe['Open']
        # testdata['High'] = testdataframe['High']
        # testdata['Low'] = testdataframe['Low']
        # testdata['Close'] = testdataframe['Close']
        # real_stock_price = testdata.iloc[:, 1:2].values
        # dataset_total = pd.concat((self.data2['Open'], testdata['Open']), axis = 0)
        # inputs = dataset_total[len(dataset_total) - len(testdata) - 60:].values
        # inputs = inputs.reshape(-1,1)
        # inputs = self.scaler.transform(inputs)
        # X_test = []

        # # for i in range(60, 235):
        # #     X_test.append(inputs[i-60:i, 0])

        # X_test = np.array(X_test)
        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        testdataframe = get_history(symbol=self.symbol, start=dt.datetime(2021,1,1), end=dt.datetime(2022,1,1))
        print("testFrame")
        
        hdfc_bank_stock_test = Stock("HDFCBANK", start_date=dt.datetime(2021, 1, 1), end_date=dt.datetime(2022, 1, 1))
        # hdfc_bank_stock.plot_history()
        real_prices = testdataframe["Open"]
        # print(real_prices)
        X_test, y_test = hdfc_bank_stock_test.preprocess()
        model = get_model()
        print(model.summary())

        self.predicted_stock_price = model.predict(X_test)
        self.predicted_stock_price = np.squeeze(self.predicted_stock_price)
        print(self.predicted_stock_price)
        print(y_test)
        # self.predicted_stock_price = np.array(self.predeicted_stock_price)
        # print("ytest = ", y_test.shape, y_test)
        # print("preds", self.predicted_stock_price.shape , self.predicted_stock_price)
        
        print("MSE =====> ", mean_squared_error(y_test, self.predicted_stock_price))


        self.predicted_stock_price = self.scaler.inverse_transform(np.expand_dims(self.predicted_stock_price, axis = 1))
        y_test = self.scaler.inverse_transform(np.expand_dims(y_test, axis=1))

        print("For real prices")

        real_prices = np.expand_dims(np.array(real_prices), axis=1)
        # print("real prices = ", real_prices)
        # print("preds", self.predicted_stock_price.shape , self.predicted_stock_price)

        # print("self.predeicted_stock_price)
        # print(real_prices.shape, self.predicted_stock_price.shape)
        print("MSE ===> ", mean_squared_error(y_test, self.predicted_stock_price))

        plt.figure(figsize=(20,10))
        plt.title('HDFCBANK Stock Price Prediction')
        plt.xlabel('Trading Day')
        plt.ylabel('HDFCBANK Stock Price')
        plt.legend()
        plt.show()




    def plot_history(self):
        plt.figure(figsize=(14,14))
        plt.plot(self.stk_data['Close'])
        plt.title('Historical Stock Value')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.show()


if __name__ == "__main__":
    start_date = dt(1999, 8, 21)
    end_date = dt(2021, 1,1)

    hdfc_bank_stock = Stock("HDFCBANK", start_date=start_date, end_date=end_date)
    hdfc_bank_stock.plot_history() 
