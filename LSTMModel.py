from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow import keras

class LSTMModel:
    def __init__(self, X_train, y_train):
        self.model = Sequential()
        self.model.add(LSTM(units = 60, return_sequences = True, input_shape = (60, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = 60, return_sequences = True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = 60, return_sequences = True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = 60))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units = 1))
        self.X_train = X_train
        self.y_train = y_train




    def train(self, epochs, batch_size):
        self.model.compile(optimizer='adam', loss="mean_squared_error")
        self.model.fit(self.X_train, self.y_train, epochs, batch_size)
        self.model.save("models/lstm_hdfcbank")


def get_model():
        return keras.models.load_model("models/lstm_hdfcbank")
