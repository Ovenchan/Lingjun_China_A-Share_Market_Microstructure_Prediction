from models_dl import *
from Dataset_mt_dl import *
import os


if __name__ == '__main__':
    
    # === 参数设置 ===
    # PARQUET_PATH = r"D:/kaggle_data"
    PARQUET_PATH = r"./data"
    TRAIN_PATH = os.path.join(PARQUET_PATH, "train")
    TEST_PATH = os.path.join(PARQUET_PATH, "test")
    PARAME_SAVE_PATH = "./model_params.pth"

    # effective_features = pl.read_csv('feature_ic_ir.csv').head(64)['Feature'].to_list()
    # FEATURE_COLS = effective_features
    FEATURE_COLS = [f"f{i}" for i in range(384)]
    

    INPUT_DIM = len(FEATURE_COLS)
    OUTPUT_DIM = 3
    HIDDEN_DIM = 128   # 减小模型容量防止过拟合
    NUM_LAYERS = 1    # 减少层数
    WINDOW_SIZE = 10
    PATCH_SIZE = 2
    EMBED_DIM = 32
    LR = 5e-4         # 降低学习率
    EPOCHS = 10        # 增加 Epoch，因为有 Early Stopping 保护
    L1_LAMBDA = 1e-5
    WEIGHT_A = 1.0
    WEIGHT_B = 0.5
    WEIGHT_C = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. 准备数据
    train_loader, val_loader = create_dataloaders(TRAIN_PATH, FEATURE_COLS, val_ratio=0.2)

    # 2. 初始化模型
    # model = QuantGRU(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    model = QuantInformer(
        input_dim=INPUT_DIM, 
        embed_dim=EMBED_DIM, 
        patch_size=PATCH_SIZE, 
        output_dim=OUTPUT_DIM, 
        topk=3
    ).to(device)
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
                
                outputs = model(x_chunk) # outputs shape: [Batch, 229, 3]
                
                # <--- 修改 2：计算多任务加权 Loss --->
                # y_chunk 也是 [Batch, 229, 3]
                # 提取前 229 个时间步，并分别计算三个标签的 MSE
                loss_A = criterion(outputs[:, :229, 0], y_chunk[:, :229, 0])
                loss_B = criterion(outputs[:, :229, 1], y_chunk[:, :229, 1])
                loss_C = criterion(outputs[:, :229, 2], y_chunk[:, :229, 2])
                
                # 加权融合
                mse_loss = WEIGHT_A * loss_A + WEIGHT_B * loss_B + WEIGHT_C * loss_C

                # # === 修改 2：仅对第一层 (GRU 的输入权重) 添加 L1 正则化 ===
                # # 在 PyTorch 的 GRU 中，`weight_ih_l0` 就是连接 Input 和 Hidden 的第一层权重矩阵
                # l1_reg = torch.tensor(0., requires_grad=True).to(device)
                # for name, param in model.named_parameters():
                #     if 'gru.weight_ih_l0' in name:
                #         # 计算绝对值的和 (L1 范数)
                #         l1_reg = l1_reg + torch.norm(param, p=1)
                
                # # 总 Loss = MSE + L1 惩罚
                # loss = mse_loss + L1_LAMBDA * l1_reg

                loss = mse_loss
                
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
                
                num_stocks = x.shape[0]
                chunk_size = 256
                batch_val_loss = 0.0
                
                # 验证集也必须采用切块策略，否则依然会 OOM
                for start_idx in range(0, num_stocks, chunk_size):
                    x_chunk = x[start_idx : start_idx + chunk_size].to(device)
                    y_chunk = y[start_idx : start_idx + chunk_size].to(device)
                    
                    outputs = model(x_chunk)
                    # v_loss = criterion(outputs[:, :229, :], y_chunk[:, :229, :])
                    v_loss = criterion(outputs[:, :229, :], y_chunk[:, :229, :])
                    
                    weight = x_chunk.shape[0] / num_stocks
                    batch_val_loss += v_loss.item() * weight
                    
                val_loss += batch_val_loss
                val_steps += 1
        
        avg_val_loss = val_loss / max(1, val_steps)
        
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
                
        #         x, y = x.to(device), y.to(device)
        #         outputs = model(x)
                
        #         # 同样只验证前 229 分钟
        #         v_loss = criterion(outputs[:, :229, :], y[:, :229, :])
        #         val_loss += v_loss.item()
        #         val_steps += 1
        
        # avg_val_loss = val_loss / max(1, val_steps)
        
        print(f"Epoch {epoch+1} Result: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # === Save Best Model ===
        if avg_val_loss < 0.99 * best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  [New Best] Val Loss improved. Saving model to {PARAME_SAVE_PATH}")
            torch.save(model.state_dict(), PARAME_SAVE_PATH)
        else:
            print(f"  Val Loss did not improve.")
            break

    # model = QuantGRU(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    # 2. 加载参数字典
    model.load_state_dict(torch.load(PARAME_SAVE_PATH, map_location=device))
    model.eval()
    generate_submission(model, TEST_PATH, FEATURE_COLS)