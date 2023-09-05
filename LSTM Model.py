class DoublePendulumLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, intermediate_size = 50):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.linear = torch.nn.Sequential(nn.Linear(hidden_size, intermediate_size), torch.nn.ReLU(inplace = True), nn.Linear(intermediate_size, output_size))

    def forward(self, x):
        h, _ = self.lstm(x)
        predictions = self.linear(h)
        return predictions
