import torch
import torch.nn as nn


class CationLSTMModel(nn.Module):
    def __init__(self, input_size, cation_size, hidden_size1=64, hidden_size2=32, output_size=1, dropout_rate=0.3):
        super(CationLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_size2 + cation_size, 16)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate/2)
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x, cation):
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)

        lstm_out = lstm2_out[:, -1, :]

        combined = torch.cat((lstm_out, cation), dim=1)

        x = self.relu(self.fc1(combined))
        x = self.dropout3(x)
        x = self.fc2(x)

        return x


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
