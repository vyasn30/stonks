from tensorflow import keras

class GRUModel:
    def __init__(self, X_train, y_train):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.GRU(units = 60, return_sequences = True, input_shape = (60, 1)))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.GRU(units = 60, return_sequences = True))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.GRU(units = 60, return_sequences = True))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.GRU(units = 60))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(units = 1))
        self.X_train = X_train
        self.y_train = y_train
        print(self.X_train.shape)
        print(self.y_train.shape)


    def train(self, epochs, batch_size):
        self.model.compile(optimizer='adam', loss="mean_squared_error")
        self.model.fit(self.X_train, self.y_train, epochs, batch_size)
        self.model.save("models/gru_hdfcbank")


def get_model():
    return keras.models.load_model("models/gru_hdfcbank")