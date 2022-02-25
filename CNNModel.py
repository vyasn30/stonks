from tensorflow import keras

class CNNModel:
    def __init__(self, X_train, y_train):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv1D(filters=60, kernel_size=2, input_shape=(60,1)))
        self.model.add(keras.layers.MaxPooling1D(pool_size=2))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Conv1D(filters=60, kernel_size=2))
        self.model.add(keras.layers.MaxPooling1D(pool_size=2))
        
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Conv1D(filters=60, kernel_size=2))
        self.model.add(keras.layers.MaxPooling1D(pool_size=2))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Conv1D(filters=60, kernel_size=2))
        self.model.add(keras.layers.MaxPooling1D(pool_size=2))
        
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(units = 1))

        self.X_train = X_train
        self.y_train = y_train
        print(self.X_train.shape)
        print(self.y_train.shape)


    def train(self, epochs, batch_size):
        self.model.compile(optimizer='adam', loss="mean_squared_error")
        self.model.fit(self.X_train, self.y_train, epochs, batch_size)
        self.model.save("models/cnn_hdfcbank")


def get_model():
    return keras.models.load_model("models/cnn_hdfcbank")