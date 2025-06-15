import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pickle

torch.manual_seed(0)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)


def create_sequences(data, window_size):
    xs, ys = [], []
    for i in range(len(data) - window_size):
        x = data[i : (i + window_size)]
        y = data[i + window_size]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


class GRU(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, dropout):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.gru = nn.GRU(
            input_size,
            hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


def model_eval(dataset):
    with open(f"./models/{dataset}_hyperparams.json") as f:
        hyperparams = json.load(f)
    lookback = hyperparams["num_inputs"]
    gru = torch.load(
        f"./models/gru_{dataset}.pt", weights_only=False, map_location=device
    )
    gru.eval()
    print(hyperparams)
    data = pd.read_csv(f"./data/raw/{dataset}.csv")["Close"].values

    Xy_train_val, Xy_test = train_test_split(data, test_size=0.2, shuffle=False)

    X_train_val, y_train_val = create_sequences(Xy_train_val, lookback)
    X_test, y_test = create_sequences(Xy_test, lookback)

    y_scaler = MinMaxScaler()
    y_train_val_scaled = y_scaler.fit_transform(y_train_val.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    X_scaler = MinMaxScaler()
    X_train_val_reshaped = X_train_val.reshape(-1, 1)
    X_test_reshaped = X_test.reshape(-1, 1)

    X_train_val_scaled = X_scaler.fit_transform(X_train_val_reshaped)
    X_test_scaled = X_scaler.transform(X_test_reshaped)

    X_train_val_scaled = X_train_val_scaled.reshape(
        X_train_val.shape[0], X_train_val.shape[1]
    )
    X_test_scaled = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1])

    predictions_scaled = []
    for i in range(len(X_test_scaled)):
        x_in = torch.tensor(X_test_scaled[i], dtype=torch.float32).to(device)
        x_in = x_in.unsqueeze(-1).unsqueeze(0)
        with torch.no_grad():
            pred = gru(x_in)
        predictions_scaled.append(pred.item())
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)

    predictions_original = y_scaler.inverse_transform(predictions_scaled).flatten()
    y_test_original = y_scaler.inverse_transform(y_test_scaled).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    mape = mean_absolute_percentage_error(y_test_original, predictions_original) * 100
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}")

    with open(f"./data/{dataset}_arima.pkl", "rb") as f:
        arima_predictions = pickle.load(f)

    plt.plot(y_test_original, label="Observed Value")
    plt.plot(predictions_original, "--", label="GRU Predictions")
    plt.plot(arima_predictions[lookback:], ":", label="ARIMA Predictions")
    plt.grid()
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(
        f"./report/images/{dataset}_predictions.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


model_eval("lzemx")
