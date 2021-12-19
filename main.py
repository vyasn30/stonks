from Dataset import Stock
from datetime import datetime as dt
from Model import Model


if __name__ == "__main__":
    hdfc_bank_stock = Stock("HDFCBANK", start_date=dt(1999, 8, 21), end_date=dt(2021, 1, 1))
    # hdfc_bank_stock.plot_history()
    X_train, y_train = hdfc_bank_stock.preprocess()
    print(X_train.shape)
    print(y_train.shape)

    # model = Model(X_train, y_train)
    # model.train(epochs = 15, batch_size=32)

    hdfc_bank_stock.predict()
