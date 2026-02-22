import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np
from tqdm import tqdm

def calculate_ic_ir(parquet_path, output_csv="feature_ic_ir.csv"):
    feature_cols = [f"f{i}" for i in range(384)]
    target_col = "LabelA"
    
    print("Scanning date IDs...")
    date_ids = pl.scan_parquet(parquet_path).select("dateid").unique().collect().to_series().sort().to_list()
    
    all_ic_records = []
    
    print("Calculating Cross-Sectional IC day by day...")
    for current_date in tqdm(date_ids):
        # try:
        #     # 核心逻辑1: 按天读取并进行基础的数据清洗
        #     df = pl.scan_parquet(parquet_path).filter(pl.col("dateid") == current_date).collect()
            
        #     if df.is_empty():
        #         continue
                
        #     df = df.with_columns([
        #         pl.col(pl.FLOAT_DTYPES).fill_nan(0).fill_null(0),
        #         pl.when(cs.float().is_infinite()).then(0).otherwise(cs.float()).name.keep()
        #     ])
            
        #     # 核心逻辑2: 按timeid分组，并行计算所有特征与LabelA的Spearman秩相关系数(即截面Rank IC)
        #     ic_exprs = [
        #         pl.corr(f, target_col, method="spearman").alias(f) 
        #         for f in feature_cols
        #     ]

        try:
            # 核心逻辑1: 按天读取数据
            df = pl.scan_parquet(parquet_path).filter(pl.col("dateid") == current_date).collect()
            
            if df.is_empty():
                continue
                
            # 分步清洗：先处理 NaN/Null，再处理 Inf。避免在同一上下文中引发列名冲突
            # df = df.with_columns(
            #     pl.col(pl.FLOAT_DTYPES).fill_nan(0).fill_null(0)
            # ).with_columns(
            #     pl.when(cs.float().is_infinite()).then(0).otherwise(cs.float()).name.keep()
            # )
            df = df.with_columns(
                cs.float().fill_nan(0).fill_null(0)
            ).with_columns(
                pl.when(cs.float().is_infinite()).then(0).otherwise(cs.float()).name.keep()
            )
            
            # 核心逻辑2: 按timeid分组，并行计算特征与LabelA的Spearman秩相关系数
            ic_exprs = [
                pl.corr(f, target_col, method="spearman").alias(f) 
                for f in feature_cols
            ]
            
            daily_ic = df.group_by("timeid").agg(ic_exprs)
            
            daily_ic_pd = daily_ic.drop("timeid").to_pandas()
            all_ic_records.append(daily_ic_pd)
            
        except Exception as e:
            print(f"Error processing date {current_date}: {e}")
            continue
            
    if not all_ic_records:
        print("No valid data processed.")
        return
        
    print("Aggregating global IC and IC_IR...")
    
    # 核心逻辑3: 拼接所有截面的IC值，计算全局的均值(Mean IC)和标准差，从而得到 IC_IR
    full_ic_df = pd.concat(all_ic_records, ignore_index=True)
    
    mean_ic = full_ic_df.mean()
    std_ic = full_ic_df.std()
    
    ic_ir = mean_ic / (std_ic + 1e-8)  # 加上平滑项防止除零异常
    
    result_df = pd.DataFrame({
        'Feature': mean_ic.index,
        'Mean_IC': mean_ic.values,
        'IC_Std': std_ic.values,
        'IC_IR': ic_ir.values
    })
    
    # 按照 IC_IR 的绝对值降序排列（因为负相关的特征通过乘以-1同样有效）
    result_df['Abs_IC_IR'] = result_df['IC_IR'].abs()
    result_df = result_df.sort_values(by='Abs_IC_IR', ascending=False).drop(columns=['Abs_IC_IR'])
    
    result_df.to_csv(output_csv, index=False)
    print(f"Top 10 Features by IC_IR:\n{result_df.head(10)}")
    print(f"Done! Saved results to {output_csv}")

if __name__ == "__main__":
    calculate_ic_ir("./data/train.parquet", "feature_ic_ir.csv")