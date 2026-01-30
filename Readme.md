# LLM Agent with System‑1 / System‑2 Thinking 
*A Dual‑Process Evaluation Framework for Reasoning‑Enhanced LLM Agents*

---

## 🔍 Overview

This repository implements a **dual‑process cognitive architecture** for Large Language Model (LLM) agents, inspired by the classical **System‑1 / System‑2** theory from cognitive science. 

- **System‑1 (S1)**: Fast, intuitive, low‑cost heuristic reasoning.
- **System‑2 (S2)**: Slow, analytical, high-fidelity deliberative pipeline ($\alpha \to \beta$) employing reflective verification.

The framework evaluates how a **Metacognitive Controller** can dynamically route tasks between S1 and S2 to optimize the **Pareto Frontier** of performance–cost trade‑offs.

---

## 🧠 Background: System‑1 vs System‑2

Dual‑process theory distinguishes between two modes of human reasoning:

| System | Characteristics | Analogy in LLMs |
|--------|----------------|----------------|
| **System‑1** | Fast, heuristic, intuitive | Small/cheap models, shallow raw likelihood |
| **System‑2** | Slow, analytical, deliberate | Large models, chain‑of‑thought, reflective audit |

Our framework operationalizes this by treating S2 invocation as a strategic **Computational Resource Allocation** problem.

---

## 🚀 Execution Workflow

To evaluate the dual-process architecture, follow these steps to generate the full inference data and run the local controller analysis.

### 1. Generate Full Inference Baseline (S1 & S2)
Run the following scripts to collect responses across all models in the registry. These scripts handle high-concurrency requests and atomic logging.

- **Run System-1 Evaluation**:
  ```bash
  python run_s1_full.py --tasks all --samples 50

  Experimental Pipeline

Run full S1 baseline

Run full S2 baseline

Compute Oracle upper bound (ideal routing)

Test local controller routing

Compare accuracy-cost Pareto frontier

📈 Key Research Questions

When does intuitive reasoning fail systematically?

Which metacognitive signals predict failure best?

How does signal reliability vary with model scale?

Can selective escalation recover most of S2 accuracy at low cost?

🔧 Supported Models

The framework is model-agnostic and supports:

edge-scale models (3B–9B) for S1

mid-weight and frontier models for S2

via unified API wrappers (OpenRouter / local inference).

📚 References

Inspired by:

Dual-Process Theory (Kahneman, Stanovich & West)

Chain-of-Thought and Self-Consistency Reasoning

LLM Cascading and Cost-Aware Routing

Metacognitive calibration in language models