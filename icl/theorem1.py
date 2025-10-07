import math, random, numpy as np, torch
from typing import Dict, List, Tuple
from dataset import TrecICLDataset
from lemma import label_conditional_logprob
# --- 1) 计算零样本 margin Δ0(x;y,y~) ---
def zero_shot_margin(model, tok, question: str, y: str, y_tilde: str, device=None) -> float:
    if device is None:
        device = next(model.parameters()).device
    # 统一 QA 风格（无任务名泄漏）
    prefix = f"Q: {question.strip()}\nA: "
    lp_y, _   = label_conditional_logprob(model, tok, prefix, y, device)
    lp_yt, _  = label_conditional_logprob(model, tok, prefix, y_tilde, device)
    return lp_y - lp_yt

# --- 2) 计算带 k-shot 的 margin Δk(p,x;y,y~) ---
def kshot_margin(model, tok, phi_task, kshot_examples: List[Tuple[str,int]], question:str, y:str, y_tilde:str, device=None) -> float:
    if device is None:
        device = next(model.parameters()).device
    # 构造 few-shot 块（建议用“Example n”分隔，避免泄漏）
    shots = []
    for i,(q, labid, *_) in enumerate(kshot_examples, 1):
        shots.append(f"### Example {i}\nQ: {q}\nA: {phi_task.verbalizers[labid]}\n")
    prompt = "".join(shots) + f"\n### Now answer the query\nQ: {question.strip()}\nA: "
    lp_y, _  = label_conditional_logprob(model, tok, prompt, y, device)
    lp_yt,_  = label_conditional_logprob(model, tok, prompt, y_tilde, device)
    return lp_y - lp_yt

# --- 3) 估计 \hat c1（独立性下界，log 形式） 和 \hat c2（每 token 概率下界，log 形式） ---
def estimate_log_c1_from_independence(stats_log_r: List[float], quantile: float=0.05) -> float:
    # stats_log_r: 你在 Assumption 2 时收集的 log 比值列表（log r<=0）
    # 取低分位（更保守）作为下界： log c1_hat
    if len(stats_log_r)==0: return -1e9
    return float(np.quantile(np.array(stats_log_r), quantile))

def estimate_log_c2(model, tok, trec, phi_task, sample_prompts: int=64, device=None, q: float=0.01) -> float:
    # 用若干条“自然 few-shot + 答案”的 prompt，收集所有下一 token 的 log prob，取低分位作为 log c2_hat
    if device is None:
        device = next(model.parameters()).device
    per_token_logs = []
    for _ in range(sample_prompts):
        exs = trec.sample_k_for_task(phi_task, k_total=2)
        shots = []
        for i,(qq, labid, *_) in enumerate(exs, 1):
            shots.append(f"### Example {i}\nQ: {qq}\nA: {phi_task.verbalizers[labid]}\n")
        txt = "".join(shots)
        with torch.no_grad():
            enc = tok(txt, return_tensors="pt")
            ids = enc.input_ids.to(device); attn = enc.attention_mask.to(device)
            out = model(ids, attention_mask=attn)
            logprobs = torch.log_softmax(out.logits, dim=-1)[:, :-1, :]           # [1,L-1,V]
            targ = ids[:, 1:]                                                     # 下一token
            lp = logprobs.gather(-1, targ.unsqueeze(-1)).squeeze(-1)              # [1,L-1]
            per_token_logs.extend(lp.flatten().tolist())
    arr = np.array(per_token_logs, dtype=np.float64)
    if len(arr)==0: return -1e9
    return float(np.quantile(arr, q))  # 非常保守的下界

# --- 4) Theorem 1 实验主函数 ---
def run_theorem1_experiment(
    trec,
    phi_task,                 # 选定 φ*（如 HUM_vs_LOC）
    model, tok,
    k_list=(1,2,4,6,8,10,12,16),
    R=50,                     # 每个 k、每个 x 上的随机 few-shot 重复次数（可小一点）
    max_tests=100,            # 测试样本数（从 φ* 的数据里取）
    device=None,
    log_r_pool=None,          # 可传入你之前 Assumption2 收集到的 log r 列表；若 None 会设一个很小的值
    quant_c1=0.05,            # c1 下界分位
    quant_c2=0.01             # c2 下界分位
) -> Dict[int, float]:
    if device is None:
        device = next(model.parameters()).device

    # 估计 θ = log c1_hat + log c2_hat  （经验 proxy）
    log_c1_hat = estimate_log_c1_from_independence(log_r_pool or [], quantile=quant_c1)
    if not np.isfinite(log_c1_hat):
        log_c1_hat = -80.0
    log_c2_hat = estimate_log_c2(model, tok, trec, phi_task, sample_prompts=32, device=device, q=quant_c2)
    theta = log_c1_hat + log_c2_hat

    # 准备测试样本（来自 φ* 的两类里各取一些）
    # 注意：对二分类任务，负类 verbalizer 就是另一个类；对 6 类任务要选“最竞争的负类”
    ds = trec.sample_k_for_task(phi_task, k_total=max_tests*2)  # (q, yid, yname)
    tests = []
    for (q, yid, yname) in ds:
        y_true = phi_task.verbalizers[yid]
        y_alt  = phi_task.verbalizers[1-yid]   # 二分类对手
        d0 = zero_shot_margin(model, tok, q, y_true, y_alt, device)
        if d0 > 0:
            tests.append((q, y_true, y_alt, d0))
        if len(tests) >= max_tests:
            break

    results = {}
    for k in k_list:
        success = 0; total = 0
        for (q, y_true, y_alt, d0) in tests:
            ok_count = 0
            for _ in range(R):
                exs = trec.sample_k_for_task(phi_task, k_total=k)   # few-shot
                dk = kshot_margin(model, tok, phi_task, exs, q, y_true, y_alt, device)
                if dk > (d0/2.0 + theta):
                    ok_count += 1
            # 统计：对每个 x，是否“以至少一半概率”满足不等式（也可以直接计频率）
            success += (ok_count / R) >= 0.5
            total += 1
        results[k] = success / max(total,1)
        print(f"[Theorem1] k={k:2d} | succ@per-x>=0.5 = {results[k]:.3f}  (theta={theta:.2f})")
    return {"theta": theta, "per_k_success": results}