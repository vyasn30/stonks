from Dataset import Stock
from datetime import datetime as dt
from RNNModel import RNNModel
from GRU_model import GRUModel
from BILSTM_model import BISLTMModel
from CNNModel import CNNModel
from LSTMModel import LSTMModel
from Hybrid_CNNLSTM import Hybrid_CNN_LSTM_Model
from Hybrid_GRU_LSTM import Hybrid_GRU_LSTM_Model


if __name__ == "__main__":
    hdfc_bank_stock = Stock("HDFCBANK", start_date=dt(2000, 8, 21), end_date=dt(2021, 1, 1))
    hdfc_bank_stock.plot_history()
    X_train, y_train = hdfc_bank_stock.preprocess()
    print(X_train.shape)
    print(y_train.shape)
    
    # model = Hybrid_GRU_LSTM_Model(X_train, y_train)
    # model.train(epochs = 15, batch_size=1)
    # model = Model(X_train, y_train)
    # model.train(epochs = 15, batch_size=32)


    hdfc_bank_stock.predict()
