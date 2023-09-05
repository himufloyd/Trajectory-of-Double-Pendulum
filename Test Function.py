def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0), desc = 'Test', leave=False, total = len(dataloader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / (len(dataloader))
