import pandas as pd
from pathlib import Path


def normalize_tf(val):
    if isinstance(val, str):
        return val.strip().lower() == 'true'
    return bool(val)


def analyze_oracle_upper_bound(results_root):
    results_root = Path(results_root)
    all_stats = []

    # 递归查找所有 _s1.csv 文件
    s1_files = list(results_root.rglob("*_s1.csv"))

    for s1_file in s1_files:
        # 1. 自动解析模型名和任务名
        # 假设格式为: {model}_{dataset}_si_s1.csv 或 {model}_{dataset}_s1.csv
        file_stem = s1_file.stem.replace("_s1", "")
        parts = file_stem.split('_')

        # 简单的拆分逻辑：假设第一个是模型，中间是任务
        # 你可以根据实际命名微调：例如 "deepseek_v3" 是模型, "si" 是任务
        if len(parts) >= 2:
            model_name = parts[0]
            task_name = "_".join(parts[1:])
        else:
            model_name = file_stem
            task_name = "unknown"

        # 2. 寻找匹配的 S2 文件
        s2_file = s1_file.parent / s1_file.name.replace("_s1.csv", "_s2.csv")
        if not s2_file.exists():
            continue

        try:
            df1 = pd.read_csv(s1_file)
            df2 = pd.read_csv(s2_file)

            # 按 task 对齐数据
            merged = pd.merge(
                df1[['task', 'T_F']],
                df2[['task', 'T_F']],
                on='task',
                how='inner',
                suffixes=('_s1', '_s2')
            )

            if merged.empty: continue

            # 转换布尔值
            merged['T_F_s1'] = merged['T_F_s1'].apply(normalize_tf)
            merged['T_F_s2'] = merged['T_F_s2'].apply(normalize_tf)

            # --- Oracle 逻辑 ---
            # Oracle 只有在 S1 错且 S2 对的时候才调用 S2
            merged['trigger_s2'] = (~merged['T_F_s1']) & (merged['T_F_s2'])
            # Oracle 最终结果：任一为真即为真
            merged['oracle_correct'] = merged['T_F_s1'] | merged['T_F_s2']

            # --- 计算指标 ---
            s1_acc = merged['T_F_s1'].mean()
            s2_acc = merged['T_F_s2'].mean()
            oracle_acc = merged['oracle_correct'].mean()

            mu = oracle_acc - s1_acc
            s2_trigger_rate = merged['trigger_s2'].mean()

            # RAR (Resource Awareness Ratio):
            # 在 Oracle 下，所有的 S2 触发都是有效的（S1错且S2对），所以理论值为 1.0
            rar = 1.0 if s2_trigger_rate > 0 else 0

            # ESC (Economic System Capability): MU / Cost
            esc = (mu / s2_trigger_rate) if s2_trigger_rate > 0 else 0

            # 3. 填入期望的格式
            all_stats.append({
                "Model": model_name,
                "Task_Name": task_name,
                "S1_Acc": round(s1_acc, 4),
                "S2_Acc": round(s2_acc, 4),
                "Oracle_Acc": round(oracle_acc, 4),
                "MU": round(mu, 4),
                "S2_Cost": round(s2_trigger_rate, 4),  # 补充指标
                "RAR": round(rar, 4),
                "ESC": round(esc, 4)
            })

        except Exception as e:
            print(f"Error processing {s1_file.name}: {e}")

    return pd.DataFrame(all_stats)


# --- 执行 ---
results_path = "Results"  # 指向你的 Results 文件夹
df_final = analyze_oracle_upper_bound(results_path)

# 按照 Model 和 Task 排序，让表格更整齐
df_final = df_final.sort_values(by=["Model", "Task_Name"])

# 保存为 CSV
df_final.to_csv("Hybrid_Oracle_Upperbound.csv", index=False)

print("✅ 理论上界分析表已生成：Hybrid_Oracle_Upperbound.csv")
print(df_final.head())