import json
import csv
import time
import yaml
import threading
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- 1. 深度标准化函数 ---
def super_normalize(text):
    """
    强力清洗：去除所有换行、特殊转义符，并将所有空白压缩为一个空格
    """
    if not text: return ""
    # 处理常见的转义字符
    text = str(text).replace('\\n', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # 压缩空格
    return " ".join(text.split()).strip()


# --- 2. 裁判逻辑 ---
def llm_judge_si(client, question, model_ans, raw_out, correct_ans):
    judge_prompt = (
        "Determine if the 'Model Answer' is factually equivalent to the 'Standard Answer'.\n"
        "Use 'Raw Output' for context. Output ONLY 'TRUE' or 'FALSE'."
    )
    user_content = f"Q: {question}\nTarget: {correct_ans}\nModel: {model_ans}\nFull: {raw_out}"
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "system", "content": judge_prompt}, {"role": "user", "content": user_content}],
            max_tokens=10, temperature=0, timeout=30
        )
        res = response.choices[0].message.content.strip().upper()
        return "True" if "TRUE" in res else "False"
    except:
        return "ERROR"


# --- 3. 主程序 ---
def main():
    # A. 加载配置
    try:
        with open("Configs/API_KEY.yaml", "r", encoding="utf-8") as f:
            api_key = yaml.safe_load(f).get("KEY")
    except:
        return print("❌ 找不到 API Key")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    results_base = Path("Results")
    si_json_path = Path("Data/si.json")

    # B. 加载标准答案库 (建立标准化索引)
    ground_truth = {}
    if not si_json_path.exists():
        return print(f"❌ 找不到标准答案文件: {si_json_path}")

    with open(si_json_path, 'r', encoding='utf-8') as j:
        data = json.load(j)
        for item in data:
            # 兼容字段名：task 或 question
            q_raw = item.get("task") or item.get("question")
            ans = item.get("correct")
            if q_raw:
                # 键名进行超强标准化
                ground_truth[super_normalize(q_raw)] = str(ans).strip()

    print(f"✅ JSON 库加载成功，共 {len(ground_truth)} 条题目")

    # C. 遍历处理 CSV
    for csv_f in results_base.rglob("*.csv"):
        if "_si_" not in csv_f.name.lower() or "_completed" in csv_f.name:
            continue

        is_s1 = "_s1" in csv_f.name.lower()
        ans_col = "s1_answer" if is_s1 else "s2_answer"
        raw_col = "s1_raw_output" if is_s1 else "s2_raw_output"

        rows = []
        with open(csv_f, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames)
            # 补全缺失列
            if "correct" not in fieldnames: fieldnames.append("correct")
            if "T_F" not in fieldnames: fieldnames.append("T_F")
            rows = list(reader)

        print(f"\n📂 正在处理: {csv_f.name}")

        tasks_to_judge = []
        match_failed_count = 0

        for row in rows:
            csv_q_raw = row.get("task", "")
            csv_q_norm = super_normalize(csv_q_raw)

            # 匹配尝试
            correct_ans = ground_truth.get(csv_q_norm)

            # 调试：如果没匹配上，打印第一条失败的原因
            if correct_ans is None:
                match_failed_count += 1
                if match_failed_count == 1:
                    print(f"⚠️ 匹配失败示例:")
                    print(f"CSV 文本: [{csv_q_norm[:50]}...]")
                    print(f"JSON 库样例: [{list(ground_truth.keys())[0][:50]}...]")
                row["correct"] = "NOT_FOUND"
                row["T_F"] = "N/A"
            else:
                row["correct"] = correct_ans
                tasks_to_judge.append(row)

        if match_failed_count > 0:
            print(f"❌ 该文件有 {match_failed_count} 行题目匹配失败，请检查文本差异！")

        # D. 执行 API 判定
        if tasks_to_judge:
            print(f"🧠 发送 {len(tasks_to_judge)} 条请求至 DeepSeek-V3...")
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_row = {
                    executor.submit(
                        llm_judge_si, client, r["task"], r[ans_col], r[raw_col], r["correct"]
                    ): r for r in tasks_to_judge
                }
                for fut in as_completed(future_to_row):
                    future_to_row[fut]["T_F"] = fut.result()

        # E. 写回文件
        with open(csv_f, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("\n✨ 任务结束")


if __name__ == "__main__":
    main()