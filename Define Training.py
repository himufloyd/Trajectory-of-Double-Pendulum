def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for i, data in tqdm(enumerate(dataloader, 0), desc = 'Train', leave=False, total = len(dataloader)):
        inputs, labels = data[0], data[1]
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        try:
            outputs = model(inputs)
        except Exception as e:
            print(data[2])
            raise RuntimeError("e")
        
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        total_loss += loss.item()
    return total_loss / (len(dataloader.dataset))
