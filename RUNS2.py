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
            all_models = yaml.safe_load(f).get("models", {})
        return api_key, all_models
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return None, None


# --- 2. S2 专用解析逻辑 ---
def parse_s2_output(text):
    """提取 Final Answer 之后的答案和置信度"""
    match = re.search(r"Final Answer:\s*(.*)", text, re.IGNORECASE)
    if match:
        full_ans_line = match.group(1).strip()
        parts = full_ans_line.split('|')
        ans = parts[0].strip()
        conf = parts[1].strip() if len(parts) > 1 else "-1"
        return ans, conf

    lines = [l for l in text.split('\n') if l.strip()]
    return (lines[-1][:100], "-1") if lines else ("PARSE_ERR", "-1")


# --- 3. S2 任务执行 (含全量指标采集) ---
def run_s2_task(task_id: int, question: str, model_id: str, client: OpenAI):
    s2_instruction = (
        "You are a deliberative System 2. Solve the question using the Alpha-Beta protocol.\n"
        "Phase 1 (Alpha): Solve the question step-by-step with deep reasoning.\n"
        "Phase 2 (Beta): Review your reasoning for logical traps or intuitive biases.\n\n"
        "Format your response as follows:\n"
        "Reasoning: <your_thought_process>\n"
        "Final Answer: [Result] | [Confidence Score 0-100]"
    )

    try:
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": s2_instruction},
                {"role": "user", "content": f"Question: {question}"}
            ],
            max_tokens=1024,
            temperature=0.7,
            timeout=60
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        raw_content = response.choices[0].message.content

        ans, conf = parse_s2_output(raw_content)

        return {
            "id": task_id,
            "task": question,
            "s2_answer": ans,
            "s2_confidence": conf,
            "s2_reasoning": raw_content.replace('\n', '  '),
            "latency_ms": latency_ms,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "s2_raw_output": raw_content.replace('\n', ' ')  # 完整原始回答
        }
    except Exception as e:
        print(f"⚠️ Task {task_id} failed: {e}")
        return None


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
            file_path = results_base / model_key / "Splits" / f"{model_key}_{dataset_name}_s2.csv"
            file_path.parent.mkdir(parents=True, exist_ok=True)

            completed_ids = set()
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader: completed_ids.add(int(row["id"]))

            todo_tasks = [t for t in tasks if t["id"] not in completed_ids]
            if not todo_tasks: continue

            print(f"🧠 Running S2: {model_key} | Dataset: {dataset_name} | Tasks: {len(todo_tasks)}")

            # 更新后的字段集
            fieldnames = [
                "id", "task", "s2_answer", "s2_confidence",
                "latency_ms", "prompt_tokens", "completion_tokens",
                "s2_reasoning", "s2_raw_output"
            ]

            with ThreadPoolExecutor(max_workers=15) as executor:
                futures = {executor.submit(run_s2_task, t["id"], t["question"], model_id, client): t["id"] for t in
                           todo_tasks}
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        with csv_lock:
                            is_new = not file_path.exists() or file_path.stat().st_size == 0
                            with open(file_path, "a", newline="", encoding="utf-8-sig") as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                if is_new: writer.writeheader()
                                writer.writerow(res)

    print("\n✨ S2 数据采集全部完成！")


if __name__ == "__main__":
    main()