# Set Path
train_path = '/content/drive/MyDrive/pendulum/Subham/Data/Train'
val_path = '/content/drive/MyDrive/pendulum/Subham/Data/Val'
test_path = '/content/drive/MyDrive/pendulum/Subham/Data/Test'
model_name = 'RNN'

if not os.path.exists('Trained_Model'):
    os.makedirs('Trained_Model')
model_param = model_params[model_name]
batch_size = model_param['batch_size']
hidden_size = model_param['hidden_size']
learning_rate = model_param['learning_rate']
num_epochs = model_param['num_epochs']

# Create dataset
train_dataset = DoublePendulumDataset(train_path)
val_dataset = DoublePendulumDataset(val_path)

# Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=model_param['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=model_param['batch_size'], shuffle=True)

# Create model, optimizer, and loss function
if model_name == 'FNN':
    model = DoublePendulum(model_param['hidden_size'], model_param['output_size'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = double_pendulum_loss
else:
    model = DoublePendulumLSTM(model_param['input_size'], model_param['hidden_size'], model_param['output_size'], model_param['num_layers'], model_param['batch_size']).double()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    

# Move model and data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# List of Loss for each epoch
epochs = [i+1 for i in range(num_epochs)]
test_epochs = []
train_losses = []
val_losses = []
test_losses = []

best_loss = float('inf')

# Train model
for epoch in tqdm(range(num_epochs), desc = "Epochs", leave = True):
    train_loss = train(model, train_dataloader, optimizer, criterion, device)
    train_losses.append(train_loss)
    val_loss = validate(model, val_dataloader, criterion, device)
    val_losses.append(val_loss)

    with open('/content/drive/MyDrive/MTP_v2/Soumadip/loss_file.txt', 'a') as loss_file:
        loss_file.write(' '.join([str(epoch), str(train_loss), str(val_loss)]) + '\n')

    total_loss = (val_loss+train_loss)*0.5
    
    if total_loss<best_loss:
        best_loss = total_loss
        best_model = model
        torch.save(best_model, model_param['model_path'])
    

plt.plot(epochs, train_losses, '-', label = "Train")
plt.plot(epochs, val_losses, '-', label = "Validation")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.savefig("/content/drive/MyDrive/MTP_v2/Soumadip/LSTM_loss.png")
plt.show()
