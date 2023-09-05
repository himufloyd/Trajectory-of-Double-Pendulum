def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0), desc = 'Validation', leave=False, total = len(dataloader)):
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(device), labels.to(device)

            try:
                outputs = model(inputs)
            except Exception as e:
                print(data[2])
                raise RuntimeError(e)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / (len(dataloader.dataset))
