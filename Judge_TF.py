import json
import csv
import re
from pathlib import Path


def sanitize_answer(text):
    """
    清洗逻辑：从文本中提取纯数字。
    支持："$1,200.50" -> 1200.5, "Result: 42" -> 42.0
    """
    if text is None: return None
    text = str(text).strip().lower()

    # 移除千分位逗号
    text = text.replace(",", "")

    # 正则提取第一个出现的数字部分（支持负号和小数点）
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def main():
    data_dir = Path("Data")
    results_base = Path("Results")

    # 1. 构建全局标准答案库 (Task -> Normalized Correct Answer)
    ground_truth = {}
    print("🔍 正在预加载 Data 目录下的标准答案...")
    for json_f in data_dir.glob("*.json"):
        # 排除简答题数据集
        if "_si_" in json_f.name.lower():
            continue

        try:
            with open(json_f, 'r', encoding='utf-8') as j:
                tasks = json.load(j)
                for item in tasks:
                    q = item.get("task") or item.get("question")
                    ans = item.get("correct")
                    if q:
                        # 存储清洗后的数值
                        ground_truth[q.strip()] = sanitize_answer(ans)
        except Exception as e:
            print(f"❌ 读取数据集 {json_f.name} 失败: {e}")

    print(f"✅ 答案库构建完成，共计 {len(ground_truth)} 条题目。")

    # 2. 遍历 Results 目录下的所有 CSV
    csv_files = list(results_base.rglob("*.csv"))

    for csv_f in csv_files:
        # 排除简答题和补全类的辅助文件
        if "_si_" in csv_f.name.lower() or "_completed" in csv_f.name.lower():
            continue

        # 识别 S1 或 S2 答案列
        ans_col = "s1_answer" if "_s1" in csv_f.name.lower() else "s2_answer"

        rows = []
        try:
            with open(csv_f, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames)

                # 动态增加 correct 和 T_F 列（如果不存在的话）
                if "correct" not in fieldnames:
                    # 建议插入到答案列后面，方便对比
                    idx = fieldnames.index(ans_col) + 1
                    fieldnames.insert(idx, "correct")
                if "T_F" not in fieldnames:
                    fieldnames.append("T_F")

                rows = list(reader)

            # 3. 逐行匹配与判分
            updated_count = 0
            for row in rows:
                q_text = row.get("task", "").strip()
                model_ans_raw = row.get(ans_col, "")

                # 获取标准答案数值
                correct_val = ground_truth.get(q_text)
                # 清洗模型给出的答案数值
                model_val = sanitize_answer(model_ans_raw)

                # 填入标准答案数值列，方便直观查看
                row["correct"] = correct_val if correct_val is not None else "N/A"

                # 数值比对逻辑
                if correct_val is not None and model_val is not None:
                    is_correct = abs(correct_val - model_val) < 1e-6
                    row["T_F"] = "True" if is_correct else "False"
                else:
                    row["T_F"] = "False"

                updated_count += 1

            # 4. 原地回写
            with open(csv_f, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"📊 已完成: {csv_f.name} | 处理行数: {updated_count}")

        except Exception as e:
            print(f"❌ 处理文件 {csv_f.name} 时出错: {e}")

    print("\n✨ 判分与标准答案补全全部完成！")


if __name__ == "__main__":
    main()