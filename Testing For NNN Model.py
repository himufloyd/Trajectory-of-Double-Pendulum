test_path = '/content/drive/MyDrive/MTP_v2/Soumadip/Data/Test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_param = model_params['RNN']
test_dataset = DoublePendulumDataset(test_path)
test_dataloader = DataLoader(test_dataset, batch_size=model_param['batch_size'], shuffle=True)
model = torch.load(model_param['model_path'])
model.eval()
output_list = []
with torch.no_grad():
    for i, data in tqdm(enumerate(test_dataloader, 0), desc = 'Test', leave=False, total = len(test_dataloader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
