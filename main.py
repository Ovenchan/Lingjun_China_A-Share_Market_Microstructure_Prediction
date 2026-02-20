from models import QuantGRU
from Dataset import *


if __name__ == '__main__':
    
    # === 参数设置 ===
    PARQUET_PATH = "./data/train.parquet"
    FEATURE_COLS = [f"f{i}" for i in range(384)]

    INPUT_DIM = 384
    HIDDEN_DIM = 64   # 减小模型容量防止过拟合
    NUM_LAYERS = 1    # 减少层数
    LR = 1e-4         # 降低学习率
    EPOCHS = 3        # 增加 Epoch，因为有 Early Stopping 保护

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. 准备数据
    train_loader, val_loader = create_dataloaders(PARQUET_PATH, FEATURE_COLS, val_ratio=0.2)

    # 2. 初始化模型
    model = QuantGRU(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    # 3. 训练循环
    for epoch in range(EPOCHS):
        # === Training Phase ===
        model.train()
        train_loss = 0
        train_steps = 0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} Training...")
        for i, (x, y) in enumerate(train_loader):
            if x.shape[0] == 1:
                x = x.squeeze(0)
                y = y.squeeze(0)
            
            if x.dim() < 3 or x.shape[0] == 0: continue
            
            optimizer.zero_grad()
            
            num_stocks = x.shape[0]
            chunk_size = 256  # 如果 64 依然 OOM，可降至 32；如果显存富余，可升至 128
            batch_loss = 0.0
            
            # --- 核心修改：对当天所有的股票进行分块处理 ---
            for start_idx in range(0, num_stocks, chunk_size):
                # 1. 切片并将当前 Chunk 移动到 GPU
                x_chunk = x[start_idx : start_idx + chunk_size].to(device)
                y_chunk = y[start_idx : start_idx + chunk_size].to(device)
                
                # 2. 前向传播
                outputs = model(x_chunk)
                
                # 3. 计算 Chunk 的 Loss
                loss = criterion(outputs[:, :229, :], y_chunk[:, :229, :])
                
                # 4. 梯度累加：按样本比例缩放 Loss，以确保等价于一次性计算全体的 Loss
                weight = x_chunk.shape[0] / num_stocks
                scaled_loss = loss * weight
                
                # 反向传播，累加梯度（注意：这一步完成后，PyTorch 会释放当前 chunk 的计算图，回收显存）
                scaled_loss.backward()
                
                batch_loss += scaled_loss.item()
            
            # 5. 执行优化器更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += batch_loss
            train_steps += 1
            
            if i % 100 == 0:
                print(f"  Step {i}, Train Loss: {batch_loss:.6f}")
        
        avg_train_loss = train_loss / max(1, train_steps)
        
        # # === Validation Phase ===
        # model.eval()
        # val_loss = 0
        # val_steps = 0
        
        # print(f"Epoch {epoch+1} Validating...")
        # with torch.no_grad():
        #     for x, y in val_loader:
        #         if x.shape[0] == 1:
        #             x = x.squeeze(0)
        #             y = y.squeeze(0)
                
        #         if x.dim() < 3 or x.shape[0] == 0: continue
                
        #         num_stocks = x.shape[0]
        #         chunk_size = 256
        #         batch_val_loss = 0.0
                
        #         # 验证集也必须采用切块策略，否则依然会 OOM
        #         for start_idx in range(0, num_stocks, chunk_size):
        #             x_chunk = x[start_idx : start_idx + chunk_size].to(device)
        #             y_chunk = y[start_idx : start_idx + chunk_size].to(device)
                    
        #             outputs = model(x_chunk)
        #             v_loss = criterion(outputs[:, :229, :], y_chunk[:, :229, :])
                    
        #             weight = x_chunk.shape[0] / num_stocks
        #             batch_val_loss += v_loss.item() * weight
                    
        #         val_loss += batch_val_loss
        #         val_steps += 1
        
        # avg_val_loss = val_loss / max(1, val_steps)
        
        # === Validation Phase ===
        model.eval()
        val_loss = 0
        val_steps = 0
        
        print(f"Epoch {epoch+1} Validating...")
        with torch.no_grad():
            for x, y in val_loader:
                if x.shape[0] == 1:
                    x = x.squeeze(0)
                    y = y.squeeze(0)
                
                if x.dim() < 3 or x.shape[0] == 0: continue
                
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                
                # 同样只验证前 229 分钟
                v_loss = criterion(outputs[:, :229, :], y[:, :229, :])
                val_loss += v_loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / max(1, val_steps)
        
        print(f"Epoch {epoch+1} Result: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # === Save Best Model ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  [New Best] Val Loss improved. Saving model to ./model_params.pth")
            torch.save(model.state_dict(), './model_params.pth')
        else:
            print(f"  Val Loss did not improve.")

    model = QuantGRU(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    # 2. 加载参数字典
    model.load_state_dict(torch.load("./model_params.pth"))
    model.eval()
    generate_submission(model, "./data/test.parquet")