class DoublePendulumDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_list = [os.path.join(path, f) for f in os.listdir(path)]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        loaded_data = np.load(self.file_list[idx], allow_pickle = True)
        data = loaded_data['data']
        inputs = data[:-1, 1:]
        targets = data[1:, 1:]
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)
        return inputs, targets
