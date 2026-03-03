from models.QuantInformer import QuantInformer
from models.NeurIF import NeurIF
from models.STHSepNet import STHSepNet
from Dataset_mt_dl_V2 import *
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def build_model(model_name, input_dim, output_dim, window_size, model_configs):
    if model_name == "QuantInformer":
        cfg = model_configs["QuantInformer"]
        return QuantInformer(
            input_dim=input_dim,
            embed_dim=cfg["embed_dim"],
            patch_size=cfg["patch_size"],
            output_dim=output_dim,
            topk=cfg["topk"],
        )

    if model_name == "NeurIF":
        cfg = model_configs["NeurIF"]
        return NeurIF(
            input_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_factors=cfg["num_factors"],
            time_steps=cfg["time_steps"],
            output_dim=output_dim,
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )

    if model_name == "STHSepNet":
        cfg = model_configs["STHSepNet"]
        return STHSepNet(
            input_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            output_dim=output_dim,
            dropout=cfg["dropout"],
            k_neighbors=cfg["k_neighbors"],
        )

    raise ValueError(f"Unsupported MODEL_NAME: {model_name}")


def unwrap_model_output(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


if __name__ == '__main__':
    
    # === 参数设置 ===
    # PARQUET_PATH = r"D:/kaggle_data"
    PARQUET_PATH = r"./data"
    TRAIN_PATH = os.path.join(PARQUET_PATH, "train")
    TEST_PATH = os.path.join(PARQUET_PATH, "test")
    
    MODEL_NAME = "NeurIF"  # 可选: QuantInformer / NeurIF / STHSepNet
    PARAME_SAVE_PATH = f"./model_params_{MODEL_NAME}.pth"

    # effective_features = pl.read_csv('feature_ic_ir.csv').head(64)['Feature'].to_list()
    # FEATURE_COLS = effective_features
    FEATURE_COLS = [f"f{i}" for i in range(384)]
    INPUT_DIM = len(FEATURE_COLS)
    OUTPUT_DIM = 3
    WINDOW_SIZE = 10
    TIME_STEPS = 239
    TARGET_STEPS = 229
    LR = 1e-5         
    EPOCHS = 15
    LAMBDA_1 = 1e-4
    LAMBDA_2 = 1e-4
    WEIGHT_A = 1.0
    WEIGHT_B = 0.5
    WEIGHT_C = 0.5

    MODEL_CONFIGS = {
        "QuantInformer": {
            "embed_dim": 64,
            "patch_size": 2,
            "topk": 3,
        },
        "NeurIF": {
            "hidden_dim": 64,
            "num_factors": 32,
            "time_steps": TIME_STEPS,
            "num_layers": 2,
            "dropout": 0.2,
        },
        "STHSepNet": {
            "hidden_dim": 64,
            "num_layers": 1,
            "dropout": 0.5,
            "k_neighbors": 5,
        },
    }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Using model: {MODEL_NAME}")

    # 1. 准备数据
    train_loader, val_loader, _, _ = create_dataloaders(TRAIN_PATH, FEATURE_COLS, val_ratio=0.2, seq_len=WINDOW_SIZE)

    # 2. 初始化模型
    model = build_model(MODEL_NAME, INPUT_DIM, OUTPUT_DIM, WINDOW_SIZE, MODEL_CONFIGS).to(device)
    model.load_state_dict(torch.load(PARAME_SAVE_PATH, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    # 3. 训练循环
    for epoch in range(EPOCHS):
        # === Training Phase ===
        model.train()
        train_loss = 0
        train_steps = 0
    
        # 用于计算训练集 R2（按第0个标签）
        train_res_sum = 0.0   # SS_res
        train_y_sum = 0.0     # sum(y)
        train_y2_sum = 0.0    # sum(y^2)
        train_n = 0           # 样本点数量
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} Training...")
        for i, (x, y) in enumerate(tqdm(train_loader)):
            if x.shape[0] == 1:
                x = x.squeeze(0)
                y = y.squeeze(0)
            
            if x.dim() < 3 or x.shape[0] == 0: continue
            
            optimizer.zero_grad()
            
            num_stocks = x.shape[0]
            chunk_size = 128  # 如果 64 依然 OOM，可降至 32；如果显存富余，可升至 128
            batch_loss = 0.0
            
            # --- 核心修改：对当天所有的股票进行分块处理 ---
            for start_idx in range(0, num_stocks, chunk_size):
                # 1. 切片并将当前 Chunk 移动到 GPU
                x_chunk = x[start_idx : start_idx + chunk_size].to(device)
                y_chunk = y[start_idx : start_idx + chunk_size].to(device)
                
                if MODEL_NAME == "NeurIF":
                    # 假设模型返回三个输出：预测值(outputs)、因子载荷(Lambda)、全局因子(F_prime)
                    outputs, Lambda, F_prime = model(x_chunk)

                    # ==========================================
                    # 1. 多任务预测主损失 (Task Losses)
                    # ==========================================
                    # 严格按照您的原始框架，计算三个 Target 的 MSE
                    loss_A = criterion(outputs[:, :TARGET_STEPS, 0], y_chunk[:, :TARGET_STEPS, 0])
                    loss_B = criterion(outputs[:, :TARGET_STEPS, 1], y_chunk[:, :TARGET_STEPS, 1])
                    loss_C = criterion(outputs[:, :TARGET_STEPS, 2], y_chunk[:, :TARGET_STEPS, 2])

                    # 加权融合得到主任务 Loss
                    loss_task = WEIGHT_A * loss_A + WEIGHT_B * loss_B + WEIGHT_C * loss_C

                    # ==========================================
                    # 2. 潜因子正交性约束 (L_orth) [cite: 35]
                    # ==========================================
                    K = F_prime.shape[-1]  # 潜因子的数量 (num_factors)
                    # 计算因子矩阵的自相关矩阵 (F^T * F)
                    F_T_F = torch.matmul(F_prime.T, F_prime) 
                    # 创建单位矩阵
                    I_K = torch.eye(K, device=F_prime.device)
                    # 计算 Frobenius 范数 (使得不同因子互相独立，提取正交 Alpha)
                    loss_orth = torch.norm(F_T_F - I_K, p='fro')

                    # ==========================================
                    # 3. 工具变量对齐约束 (L_inst) [cite: 35]
                    # ==========================================
                    # 计算横截面均值
                    Lambda_mean = Lambda.mean(dim=0, keepdim=True)
                    # 计算因子载荷偏离横截面均值的 L2 范数 (形状: [N, T])
                    Lambda_dev = torch.norm(Lambda - Lambda_mean, dim=-1)

                    # 注意：X_dev 应该是由您的底层高频微观特征计算出的横截面偏差
                    # 这里暂时使用随机张量占位，您需要将其替换为类似: 
                    # torch.norm(X_features - X_features.mean(dim=0, keepdim=True), dim=-1)
                    X_dev = torch.rand_like(Lambda_dev) 

                    # 工具惩罚：迫使因子载荷的分布结构逼近底层微观特征的分布结构
                    loss_inst = torch.mean((Lambda_dev - X_dev) ** 2)

                    # 最终的总损失
                    loss = loss_task + LAMBDA_1 * loss_orth + LAMBDA_2 * loss_inst
                else:
                    model_output = model(x_chunk)
                    outputs = unwrap_model_output(model_output)  # outputs shape: [Batch, TimeSteps, OUTPUT_DIM]
                    
                    # <--- 修改 2：计算多任务加权 Loss --->
                    # y_chunk 也是 [Batch, 229, 3]
                    # 提取前 229 个时间步，并分别计算三个标签的 MSE
                    loss_A = criterion(outputs[:, :TARGET_STEPS, 0], y_chunk[:, :TARGET_STEPS, 0])
                    loss_B = criterion(outputs[:, :TARGET_STEPS, 1], y_chunk[:, :TARGET_STEPS, 1])
                    loss_C = criterion(outputs[:, :TARGET_STEPS, 2], y_chunk[:, :TARGET_STEPS, 2])
                    
                    # 加权融合
                    mse_loss = WEIGHT_A * loss_A + WEIGHT_B * loss_B + WEIGHT_C * loss_C

                    loss = mse_loss
                    
                # 4. 梯度累加：按样本比例缩放 Loss，以确保等价于一次性计算全体的 Loss
                weight = x_chunk.shape[0] / num_stocks
                scaled_loss = loss * weight
                
                # 反向传播，累加梯度（注意：这一步完成后，PyTorch 会释放当前 chunk 的计算图，回收显存）
                scaled_loss.backward()
                
                batch_loss += scaled_loss.item()

                # 统计训练 R2（第0列）
                y_true_r2 = y_chunk[:, :TARGET_STEPS, 0]
                y_pred_r2 = outputs[:, :TARGET_STEPS, 0]
                train_res_sum += ((y_true_r2 - y_pred_r2) ** 2).sum().item()
                train_y_sum += y_true_r2.sum().item()
                train_y2_sum += (y_true_r2 ** 2).sum().item()
                train_n += y_true_r2.numel()
            
            # 5. 执行优化器更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += batch_loss
            train_steps += 1
            
            if i % 100 == 0:
                print(f"  Step {i}, Train Loss: {batch_loss:.6f}")
        
        avg_train_loss = train_loss / max(1, train_steps)

        # 计算训练 R2
        train_den = train_y2_sum - (train_y_sum ** 2) / max(1, train_n)
        train_r2 = 1.0 - train_res_sum / train_den if train_den > 1e-12 else float("nan")
        
        # === Validation Phase ===
        model.eval()
        val_loss = 0
        val_steps = 0

        # 用于计算验证集 R2（按第0个标签）
        val_res_sum = 0.0
        val_y_sum = 0.0
        val_y2_sum = 0.0
        val_n = 0
        
        print(f"Epoch {epoch+1} Validating...")
        with torch.no_grad():
            for x, y in tqdm(val_loader):
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
                    
                    model_output = model(x_chunk)
                    outputs = unwrap_model_output(model_output)
                    v_loss = criterion(outputs[:, :TARGET_STEPS, :], y_chunk[:, :TARGET_STEPS, :])
                    # v_loss = criterion(outputs[:, :TARGET_STEPS, 0], y_chunk[:, :TARGET_STEPS, 0])
                    
                    weight = x_chunk.shape[0] / num_stocks
                    batch_val_loss += v_loss.item() * weight
                    
                    # 统计验证集 R2（第0列）
                    y_true_r2 = y_chunk[:, :TARGET_STEPS, 0]
                    y_pred_r2 = outputs[:, :TARGET_STEPS, 0]
                    val_res_sum += ((y_true_r2 - y_pred_r2) ** 2).sum().item()
                    val_y_sum += y_true_r2.sum().item()
                    val_y2_sum += (y_true_r2 ** 2).sum().item()
                    val_n += y_true_r2.numel()

                val_loss += batch_val_loss
                val_steps += 1
        
        avg_val_loss = val_loss / max(1, val_steps)
        
        # 计算验证集 R2
        val_den = val_y2_sum - (val_y_sum ** 2) / max(1, val_n)
        val_r2 = 1.0 - val_res_sum / val_den if val_den > 1e-12 else float("nan")
        
        print(f"Epoch {epoch+1} Result: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        print(f"Epoch {epoch+1} Train R2 = {train_r2:.6f}, Val R2 = {val_r2:.6f}")
        
        # === Save Best Model ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  [New Best] Val Loss improved. Saving model to {PARAME_SAVE_PATH}")
            torch.save(model.state_dict(), PARAME_SAVE_PATH)
        else:
            print(f"  Val Loss did not improve.")


    # 2. 加载参数字典
    model.load_state_dict(torch.load(PARAME_SAVE_PATH, map_location=device))
    model.eval()
    generate_submission_with_scale(model, TEST_PATH, FEATURE_COLS, seq_len=WINDOW_SIZE, train_parquet_path = TRAIN_PATH)