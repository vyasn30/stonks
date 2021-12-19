from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras

class Model:
    def __init__(self, shape):
        self.model = Sequential()
        self.model.add(LSTM(units = 50, return_sequences = True, input_shape = (shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = 50, return_sequences = True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = 50, return_sequences = True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = 50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units = 1))
        self.X_train = None
        self.y_train = None




    def train(self, X_train, y_train, epochs, batch_size):
        self.X_train = X_train
        self.y_train = y_train
        self.model.compile(optimizer='adam', loss="mean_squared_error")
        self.model.fit(self.X_train, self.y_train, epochs, batch_size)
        self.model.save("models/lstm_hdfcbank")


    def get_model(self):
        return keras.models.load_model("models/lstm_hdfcbank")