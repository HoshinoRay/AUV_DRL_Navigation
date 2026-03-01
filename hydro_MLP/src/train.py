import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time

# 引入新写的 Manager
from dataset import HydroDataManager
from model import DeepHydroMLP # 假设你的 model.py 保持不变

# ================= 配置参数 =================
CSV_PATH = '/home/ray/Disk_ext/DeepSim_RL/hydro_MLP/data/phy_processed/mission_log_processed_Physics.csv' 
BATCH_SIZE = 512        # 80万数据，建议开大 Batch Size 加速训练，同时梯度更稳
LR = 0.001              
EPOCHS = 100            # 增加轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = '../models/'
LOG_DIR = '../logs/'
# ===========================================

def main():
    print(f"Training on device: {DEVICE}")
    
    # 1. 使用 DataManager 获取数据
    # 这里自动完成了：读取 -> 打乱(Shuffle) -> 切分 -> 归一化 -> 封装
    data_manager = HydroDataManager(CSV_PATH, save_dir=SAVE_DIR)
    train_loader, val_loader, test_loader = data_manager.get_dataloaders(
        batch_size=BATCH_SIZE, 
        val_split=0.1,  # 10% 做验证
        test_split=0.05 # 5% 做最终测试 (完全没见过的数据)
    )
    
    # 2. 初始化模型
    model = DeepHydroMLP(input_dim=12, output_dim=6).to(DEVICE)
    
    # 优化器设置：增加 Weight Decay 防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # 学习率调度器：如果 Loss 不降了，自动减小学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    criterion = nn.MSELoss()
    
    writer = SummaryWriter(log_dir=LOG_DIR + f'run_{int(time.time())}')

    # 3. 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # --- Logging ---
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_hydro_model.pth'))
            print("  >>> Best model saved!")

    print("Training Finished!")
    writer.close()
    
    # (可选) 训练结束后，可以用 test_loader 再跑一次纯净测试
    # evaluate_on_test(model, test_loader) 

if __name__ == '__main__':
    main()