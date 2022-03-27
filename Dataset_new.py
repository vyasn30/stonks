from lightgbm import train
import pandas as pd
import glob
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
# from LSTM_2 import get_model
# from GRU_model2 import get_model
from Hybrid_GRU_LSTM_2 import get_model
from nsepy import get_history
import datetime as dt

class Dataset:
	def __init__(self, symbol, start_date, end_date):
		self.symbol = 'HDFCBANK'
		self.start_date = start_date
		self.end_date = end_date
		self.stk_data = get_history(symbol=self.symbol, start=self.start_date, end=self.end_date)
		# print("/n fetched data")
		# print(self.stk_data)
		# print(self.stk_data.index)
		
		self.stk_data['Date'] = self.stk_data.index
		self.stk_data = self.correction(self.stk_data)
		self.data2 = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
		self.data2['Date'] = self.stk_data['Date']
		self.data2['Open'] = self.stk_data['Open']
		self.data2['High'] = self.stk_data['High']
		self.data2['Low'] = self.stk_data['Low']
		self.data2['Close'] = self.stk_data['Close']
		self.data2["Total Traded Quantity"] = self.stk_data['Volume']
		self.scaler = None
		self.predeicted_stock_price = None
		self.X, self.y = self.preprocess()
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33)	
		# print(self.data2)
		self.y_pred = None
		# print(self.data2)
# self.stk_data = self.load()
		# self.stk_data['Date'] = self.stk_data.index
		# self.data2 = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])

		# self.data2['Date'] = self.stk_data['Date']
		# self.data2['Open'] = self.stk_data['Open Price']
		# self.data2['High'] = self.stk_data['High Price']
		# self.data2['Low'] = self.stk_data['Low Price']
		# self.data2['Close'] = self.stk_data['Close Price']
		# self.data2['Total Traded Quantity'] = self.stk_data['Total Traded Quantity']

		# self.scaler = None
		# self.y_pred = None
		# self.predicted_stock_price = None
		# self.X, self.y = self.preprocess()
		# self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33)
		# print(self.data2)
	def correction(self, stk_data):
		def myfunc(x,date):
			if date < dt.date(2019, 10, 30):
				return (x/10) 
			else:
				return (x)

		print("\n\n In correction")
		print(stk_data)

		for index, row in stk_data.iterrows():
			# print(type(row["Date"]))
			
			row = row.copy()

			# row["Open"] = myfunc(row["Open"], row["Date"])
			stk_data.at[index, "Open"] = myfunc(row["Open"], row["Date"])
			# row["Close"] = myfunc(row["Close"], row["Date"])
			stk_data.at[index, "High"] = myfunc(row["High"], row["Date"])
			
			stk_data.at[index, "Close"] = myfunc(row["Close"], row["Date"])
			
			stk_data.at[index, "Low"] = myfunc(row["Low"], row["Date"])
			
			
			# row["High"] = myfunc(row["High"], row["Date"])
			
			# row["Low"] = myfunc(row["Low"], row["Date"])
			
		# stk_data['Open']=stk_data['Open'].apply(lambda x: x/10 if dt(stk_data["Date"]) < dt('30-10-2019') else '')
		
		# stk_data['Close']=stk_data['Close'].apply(lambda x: x/10 if dt(stk_data["Date"]) < dt('30-10-2019') else '')
		
		# stk_data['High']=stk_data['High'].apply(lambda x: x/10 if dt(stk_data["Date"]) < dt('30-10-2019') else '')
		
		# stk_data['Low']=stk_data['Low'].apply(lambda x: x/10 if dt(stk_data["Date"]) < dt('30-10-2019') else '')
		
		
	
			
		print(stk_data)
  
		# for val in stk_data:
			# print(val)
			# print("\n")
  
		return stk_data

	def preprocess(self):
		train_set = self.data2.iloc[:, 1:].values
		# print(train_set)
		# train_set = self.correction(train_set)
		self.scaler = MinMaxScaler()
		train_set_scaled = self.scaler.fit_transform(train_set)

		X = []
		y = []

		for i in range(30, len(train_set_scaled)):
			X.append(train_set_scaled[i-30:i])
			y.append(train_set_scaled[i])

		X, y = np.array(X), np.array(y)

		print("X_train shape", X.shape)
		print(X)
		print("Y_train shape", y.shape)
		print(y)
	
		return X, y



	def predict(self):
		model = get_model()
		self.y_pred = model.predict(self.X_test)
		predicted_frame_transformed = self.scaler.inverse_transform(self.y_pred)

		# print(predicted_frame)
		
		return self.y_pred, predicted_frame_transformed
		
	def find_mse(self):

		open_price_test = [val[0] for val in self.y_test]
		open_price_pred = [val[0] for val in self.y_pred]

		return mean_squared_error(open_price_test, open_price_pred)

		
	def load(self):
		path = r'data'
		file_list = glob.glob(path + "/*.csv")
		li = []

		for filename in file_list:
			df = pd.read_csv(filename, index_col=None, header=0)
			li.append(df)

		data =  pd.concat(li, axis=0, ignore_index=True)
		print(data)
		return data

if __name__ == "__main__":
	frame = Dataset()
	
