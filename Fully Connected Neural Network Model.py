class DoublePendulum(nn.Module):
    def __init__(self, hidden_size=[64, 128, 256, 512], output_size = 4):
        super(DoublePendulum, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.fc5 = nn.Linear(hidden_size[3], hidden_size[2])
        self.fc6 = nn.Linear(hidden_size[2], hidden_size[1])
        self.fc7 = nn.Linear(hidden_size[1], hidden_size[0])
        self.fc8 = nn.Linear(hidden_size[0], output_size)

    def forward(self, x):
        x = x.to(self.fc1.weight.dtype)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.fc8(x)
        return x
