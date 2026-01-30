import json
import csv
import time
import re
import yaml
import threading
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- 1. 配置加载 ---
def load_config():
    try:
        with open("Configs/API_KEY.yaml", "r", encoding="utf-8") as f:
            api_key = yaml.safe_load(f).get("KEY")
        with open("Configs/models.yaml", "r", encoding="utf-8") as f:
            models_data = yaml.safe_load(f)
            all_models = models_data.get("models", {})
        return api_key, all_models
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return None, None


# --- 2. 核心解析逻辑 ---
def parse_s1_output(raw_text):
    text = raw_text.strip()
    text = re.sub(r'```json\s*|```', '', text).strip()
    ans, conf = "PARSE_ERR", -1
    try:
        data = json.loads(text)
        ans = str(data.get("answer", "PARSE_ERR"))
        conf = int(data.get("confidence", -1))
    except:
        match = re.search(r'(.*)\|\s*(\d+)', text)
        if match:
            ans = match.group(1).replace('[', '').replace(']', '').strip()
            try:
                conf = int(match.group(2))
            except:
                conf = -1
    return ans, conf


# --- 3. S1 任务执行 (增强版：记录消耗与延迟) ---
def run_s1_task(task_id: int, question: str, model_id: str, client: OpenAI):
    system_instruction = (
        "You are an intuitive S1 engine. Respond instantly and concisely.\n"
        "Format: {\"answer\": \"your_ans\", \"confidence\": 0-100}"
    )

    samples = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    start_wall_time = time.perf_counter()  # 记录总耗时开始

    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": question}
                ],
                max_tokens=80,
                temperature=0.3,
                timeout=20
            )
            raw = response.choices[0].message.content
            ans, conf = parse_s1_output(raw)

            # 累加 Token 消耗
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens

            samples.append({"ans": ans, "conf": conf, "raw": raw})
        except:
            continue

    if not samples:
        return None

    # 计算总延迟 (毫秒)
    total_latency_ms = int((time.perf_counter() - start_wall_time) * 1000)

    valid_answers = [s['ans'] for s in samples if s['ans'] != "PARSE_ERR"]
    unique_answers = set(valid_answers)
    primary = samples[0]

    return {
        "id": task_id,
        "task": question,
        "s1_answer": primary['ans'],
        "s1_confidence": primary['conf'],
        "consistency_entropy": len(unique_answers),
        "latency_ms": total_latency_ms,  # 核心指标 1：总耗时
        "prompt_tokens": total_prompt_tokens,  # 核心指标 2：输入 Token
        "completion_tokens": total_completion_tokens,  # 核心指标 3：输出 Token
        "s1_raw_output": primary['raw'].replace('\n', ' '),
        "samples_count": len(samples)
    }


# --- 4. 主控流程 ---
csv_lock = threading.Lock()


def main():
    api_key, all_models = load_config()
    if not api_key: return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    data_dir = Path("Data")
    results_base = Path("Results")

    for json_f in data_dir.glob("*.json"):
        dataset_name = json_f.stem
        with open(json_f, 'r', encoding='utf-8') as j:
            tasks = [{"id": i, "question": (item.get("task") or item.get("question"))}
                     for i, item in enumerate(json.load(j))]

        for model_key, info in all_models.items():
            model_id = info['id']
            file_path = results_base / model_key / "Splits" / f"{model_key}_{dataset_name}_s1.csv"
            file_path.parent.mkdir(parents=True, exist_ok=True)

            completed_ids = set()
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader: completed_ids.add(int(row["id"]))

            todo_tasks = [t for t in tasks if t["id"] not in completed_ids]
            if not todo_tasks: continue

            print(f"🚀 Running S1: {model_key} | Dataset: {dataset_name} | Tasks: {len(todo_tasks)}")

            # 这里的 fieldnames 必须包含新增的三个指标
            fieldnames = [
                "id", "task", "s1_answer", "s1_confidence",
                "consistency_entropy", "latency_ms", "prompt_tokens",
                "completion_tokens", "s1_raw_output", "samples_count"
            ]

            with ThreadPoolExecutor(max_workers=15) as executor:
                future_to_id = {executor.submit(run_s1_task, t["id"], t["question"], model_id, client): t["id"] for t in
                                todo_tasks}

                for future in as_completed(future_to_id):
                    res = future.result()
                    if res:
                        with csv_lock:
                            is_new = not file_path.exists() or file_path.stat().st_size == 0
                            with open(file_path, "a", newline="", encoding="utf-8-sig") as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                if is_new: writer.writeheader()
                                writer.writerow(res)

    print("\n✨ S1 数据采集全部完成（含消耗与延迟指标）！")


if __name__ == "__main__":
    main()