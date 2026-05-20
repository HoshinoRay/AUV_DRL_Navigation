"""
Train DeepHydroMLP on the processed physics CSV.

Usage (from hydro_MLP/):
    python src/train.py

Override the CSV path via the CSV_PATH constant below or pass it as an
environment variable: CSV_PATH=/path/to/data.csv python src/train.py
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import HydroDataManager
from model import DeepHydroMLP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = os.environ.get(
    'CSV_PATH',
    '../data/phy_processed/mission_log_processed_Physics.csv',
)
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = '../models/'
LOG_DIR = '../logs/'
# ---------------------------------------------------------------------------


def main():
    print(f"Training on: {DEVICE}")

    data_manager = HydroDataManager(CSV_PATH, save_dir=SAVE_DIR)
    train_loader, val_loader, _ = data_manager.get_dataloaders(
        batch_size=BATCH_SIZE,
        val_split=0.1,
        test_split=0.05,
    )

    model = DeepHydroMLP(input_dim=12, output_dim=6).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, f'run_{int(time.time())}'))

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                val_loss += criterion(model(inputs), targets).item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]  "
            f"Train: {avg_train_loss:.6f}  Val: {avg_val_loss:.6f}  LR: {current_lr:.6f}"
        )
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_hydro_model.pth'))
            print("  -> Best model saved.")

    print("Training complete.")
    writer.close()


if __name__ == '__main__':
    main()
