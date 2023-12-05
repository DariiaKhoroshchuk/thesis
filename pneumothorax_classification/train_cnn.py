import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import CustomDataset, train_transform, validation_transform
from models import SmallCNN
from constants import CHECKPOINT_DIR

lr = 0.0001
batch_size = 8

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
print(sum(p.numel() for p in model.parameters()))


# Training function
def train(epoch, model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        running_loss += loss.item()

    accuracy = 100. * correct / total
    print(f"[Epoch {epoch}] loss: {running_loss / len(dataloader)}, Accuracy: {accuracy}%")
    return running_loss / len(dataloader), accuracy


# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f"Validation loss: {val_loss / len(dataloader)}, Accuracy: {accuracy}%")
    return val_loss / len(dataloader), accuracy


if __name__ == '__main__':
    t0 = time.time()
    # load training data
    train_dataset = CustomDataset('train', augmentations=train_transform, kind=None)
    labels = train_dataset.labels
    num_positive = sum(labels)
    total_samples = len(labels)
    print(f'training positive ratio: {num_positive / total_samples}')
    positive_weight = (total_samples - num_positive) / num_positive
    negative_weight = num_positive / (total_samples - num_positive)
    weights = [negative_weight if label == 0 else 1 for label in labels]

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    # Load validation data
    val_dataset = CustomDataset('validation', augmentations=validation_transform, kind=None)
    labels = val_dataset.labels
    num_positive = sum(labels)
    total_samples = len(labels)
    print(f'validation positive ratio: {num_positive / total_samples}')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_epochs = 200
    patience = 25
    best_val_loss = float('inf')

    save_dir = f'{CHECKPOINT_DIR}/cnn_norm_{lr}_{batch_size}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    val_loss_path = os.path.join(save_dir, 'val_losses.txt')
    time_dataset = time.time() - t0
    print(f"Time for dataset loading: {time_dataset}")
    since = time.time()
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_accuracy = train(epoch, model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        time_epoch = time.time() - t0
        with open(val_loss_path, 'a') as f:
            f.write(f"Epoch {epoch}:\n train loss: {train_loss}, train acc: {train_accuracy}, val loss: {val_loss}, val acc: {val_accuracy}, {time_epoch // 60:.0f}m {time_epoch % 60:.0f}s\n")

        checkpoint_path = os.path.join(save_dir, f'checkpoint_{epoch}.pth')
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Best val loss: {best_val_loss}")
                print("Early stopping triggered!")
                time_elapsed = time.time() - since
                print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed %60:.0f}s')
                break
