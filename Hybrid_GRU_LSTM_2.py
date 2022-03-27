from tensorflow import keras

class Hybrid_GRU_LSTM_model:
    def __init__(self, X_train, y_train):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.GRU(units = 1, return_sequences = True, input_shape = (30, 5)))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.GRU(units = 30, return_sequences = True))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.GRU(units = 30, return_sequences = True))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.LSTM(units = 30, return_sequences = True))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.LSTM(units = 30))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(units = 5))
        self.X_train = X_train
        self.y_train = y_train
        print(self.X_train.shape)
        print(self.y_train.shape)


    def train(self, epochs, batch_size):
        self.model.compile(optimizer='adam', loss="mean_squared_error")
        self.model.fit(self.X_train, self.y_train, epochs, batch_size)
        self.model.save("models/hybrid_gru_lstm_hdfcbank_multiparam")


def get_model():
    return keras.models.load_model("models/hybrid_gru_lstm_hdfcbank_multiparam")