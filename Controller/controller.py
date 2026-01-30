# controller.py
from S2.S2 import run_s2
from S1.S1 import run_s1

def should_call_s2(s1_out, thresholds):
    """
    决策函数：根据 S1 输出判断是否需要调用 S2
    """
    return not (
        s1_out["confidence"] > thresholds.get("conf", 0.8)
        and s1_out["perplexity"] < thresholds.get("ppl", 5.0)
        and s1_out["self_consistency"] > thresholds.get("sc", 0.8)
    )
