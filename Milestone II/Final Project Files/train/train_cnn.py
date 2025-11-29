import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

from data.data import set_seed, get_data_dir, load_metadata, get_dataloaders
from models.cnn_models import EffNetB0, EfficientNetB0_CBAM, EfficientNetB2_CBAM

# Set seed
set_seed(42)

# Hyperparameters/Options
batch_size = 32
num_epochs = 10
learning_rate = .0001
dropout = 0.2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model_by_name(name, **kwargs):
    """Factory function to create models"""
    if name == "b0":
        return EffNetB0(
            pretrained=kwargs.get("pretrained", True)
        )
    elif name == "b0_cbam":
        return EfficientNetB0_CBAM(
            num_classes=kwargs.get("num_classes", 2),
            pretrained=kwargs.get("pretrained", True),
            dropout=kwargs.get("dropout", 0.2)
        )
    elif name == "b2_cbam":
        return EfficientNetB2_CBAM(
            num_classes=kwargs.get("num_classes", 2),
            pretrained=kwargs.get("pretrained", True),
            dropout=kwargs.get("dropout", 0.2)
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def plot_training_history(history, title="Training History"):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4):
    """
    Train and evaluate a model.
    
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        scheduler.step()

    return history

def main(args):
    print("Using device:", device)

    # Load data
    data_dir = get_data_dir()
    meta_df = load_metadata(data_dir)
    
    # Get dataloaders - returns 4 loaders
    train_loader, train_strong_loader, val_loader, test_loader = get_dataloaders(
        meta_df, 
        data_dir,
        batch_size=args.batch_size
    )
    
    # Select which training loader to use based on strong_aug flag
    if args.strong_aug:
        active_train_loader = train_strong_loader
        print("Using HEAVY augmentation (CutMix + ColorJitter)")
    else:
        active_train_loader = train_loader
        print("Using LIGHT augmentation (basic)")

    # Get model
    model = get_model_by_name(
        args.model, 
        pretrained=args.pretrained, 
        dropout=args.dropout,
        num_classes=2
    )
    model = model.to(device)

    # Train model
    history = train_model(
        model, 
        active_train_loader, 
        val_loader, 
        num_epochs=args.epochs, 
        lr=args.lr
    )

    # Plot results
    plot_training_history(history, title=f"{args.model}")
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN model")
    parser.add_argument("--model", type=str, 
                       choices=["b0", "b0_cbam", "b2_cbam"], 
                       default="b0_cbam",
                       help="Model architecture to use")
    parser.add_argument("--pretrained", action="store_true", 
                       help="Use pretrained ImageNet weights")
    parser.add_argument("--strong_aug", action="store_true", 
                       help="Use heavy augmentation (CutMix + ColorJitter)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()
    main(args)