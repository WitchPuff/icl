from dataset import TrecICLDataset
from typing import List, Dict, Iterable, Optional, Tuple
import math
import random
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

STYLE = "qa"  # ["instruction", "qa", "minimal"]
K_LIST = tuple(list(range(0, 5)) + list(range(6, 16 + 1, 2)))  # k values to test
EPSILON = 1e-2  # concentration threshold
print("Using prompt style:", STYLE)
device = "cuda" if torch.cuda.is_available() else "cpu"


def to_device(model) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return device


def text_logprob(model, tok, text: str, device='cuda') -> float:
    """整段文本的对数似然 log P(text)。"""
    with torch.no_grad():
        enc = tok(text, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        outputs = model(input_ids, attention_mask=attn, labels=input_ids)
        n_tokens = attn.sum().item()
        return -outputs.loss.item() * n_tokens


def label_conditional_logprob(
    model,
    tok,
    prefix: str,
    label_text: str,
    device: torch.device,
):
    label_text = (label_text or "").rstrip("\n") + '\n'
    with torch.no_grad():
        full = prefix + label_text
        enc_full = tok(full, return_tensors="pt")
        enc_pref = tok(prefix, return_tensors="pt")
        ids_full = enc_full.input_ids.to(device)
        ids_pref = enc_pref.input_ids.to(device)
        n_pref = ids_pref.shape[1]
        attn = torch.ones_like(ids_full, device=device)
        outputs = model(ids_full, attention_mask=attn)
        logits = outputs.logits  # [1, L, V]
        logprobs = torch.log_softmax(logits, dim=-1)
        next_token_logprobs = logprobs[:, :-1, :]                         # [1, L-1, V]
        gather = next_token_logprobs.gather(-1, ids_full[:, 1:].unsqueeze(-1)).squeeze(-1)  # [1, L-1]
        label_lp = gather[:, n_pref-1:].sum().item()
        n_label_tokens = (ids_full[:, n_pref:].numel())
        return label_lp, n_label_tokens


import random, string

def random_sep(length=4):
    """生成一个随机的分隔符，例如 '### SEP_XQZT ###'"""
    tag = "".join(random.choices(string.ascii_uppercase, k=length))
    return f"\n### SEP_{tag} ###\n"


def make_trec_prompt(
    question="",
    label_name=None,
    style="qa",          # ["instruction", "qa", "minimal"]
    include_label=True,
    add_random_sep=True  # 是否添加随机分隔符，
) -> str:
    """构造 TREC few-shot 示例的 prompt。"""
    sep = random_sep() if add_random_sep else "\n"
    prefix = ""

    if include_label and label_name is not None:
        prefix = f"{sep}### Example:{sep}"

    if style == "instruction":
        prefix += f"Question: {question.strip()}\nPlease answer with the correct category."
        if include_label and label_name is not None:
            return f"{prefix} {label_name.strip().lower()}\n\n"
        else:
            return prefix + " "

    elif style == "qa":
        prefix += f"Q: {question.strip()}\nA:"
        if include_label and label_name is not None:
            return f"{prefix} {label_name.strip().lower()}\n\n"
        else:
            return prefix + " "

    elif style == "minimal":
        prefix += f"{question.strip()}\nCategory:"
        if include_label and label_name is not None:
            return f"{prefix} {label_name.strip().lower()}\n\n"
        else:
            return prefix + " "

    else:
        raise ValueError(f"Unknown style: {style}")


def test_trec(
    trec,
    model,
    tok,
    max_eval=None,
    k_list=(0, 8, 16, 64),
    flip_prob=0,
    device='cuda'
) -> Dict[int, float]:
    task = trec.get_original_task()
    if 0 not in k_list:
        k_list = (0,) + tuple(k_list)
    correct = [0 for _ in k_list]
    total = [0 for _ in k_list]
    if max_eval is not None:
        original = trec.sample_k_for_task(task, k_total=max_eval, flip_prob=flip_prob)
    else:
        original = trec.get_original_dataset()

    for i, k in tqdm(enumerate(k_list), total=len(k_list), desc="Evaluating TREC"):
        print(f"  k = {k}")
        for (text, _lid, name) in original:
            prefix = task.instr_blind + "\n\n"
            if k > 0:
                support = trec.sample_k_for_task(task, k_total=k, flip_prob=flip_prob)
                for (sq, _sy, slab) in support:
                    prefix += make_trec_prompt(sq, label_name=slab, style=STYLE, include_label=True)
            prefix += make_trec_prompt(text, label_name=None, style=STYLE, include_label=False)
            scores = []
            for cname in trec.verbalizers.values():
                lp, _ = label_conditional_logprob(model, tok, prefix, cname.lower(), device)
                scores.append((lp, cname))
            pred = max(scores, key=lambda x: x[0])[1]
            if pred == name:
                correct[i] += 1
            total[i] += 1
    return {k: c / t if t > 0 else 0 for c, t, k in zip(correct, total, k_list)}


# -----------------------------------------------------------
# 1) Assumption 2：独立性下界 c1 的经验估计（返回统计 + 全部 log_r 池）
# -----------------------------------------------------------

def estimate_c1_independence(
    trec,
    task,
    model,
    tok,
    trials=100,
    k_list=(1, 2, 8, 16, 64),
    blind=False,
) -> Dict[int, Dict[str, float]]:
    device = next(model.parameters()).device
    ratios = {k: [] for k in k_list if k > 0}
    pool_log_rs: List[float] = []
    print(f"[Step 1] Estimating independence ratios for task {task.name}, blind={blind}...")
    results = {}
    for ki in k_list:
        if ki == 0:
            continue
        for _ in range(trials):
            samp1 = trec.sample_k_for_task(task, k_total=ki)
            samp2 = trec.sample_k_for_task(task, k_total=ki)
            s1 = (task.instr_blind if blind else task.instr) + "\n\n"
            for ex in samp1:
                if len(ex) == 3:
                    q, y, lab = ex
                else:
                    q, y = ex
                    lab = task.verbalizers[y]
                s1 += make_trec_prompt(q, label_name=lab, style=STYLE, include_label=True)
            s2 = ""
            for ex in samp2:
                if len(ex) == 3:
                    q, y, lab = ex
                else:
                    q, y = ex
                    lab = task.verbalizers[y]
                s2 += make_trec_prompt(q, lab, style=STYLE, include_label=True)

            lp_s1n = text_logprob(model, tok, s1 + "\n", device)
            lp_s2 = text_logprob(model, tok, s2, device)
            lp_s1s2 = text_logprob(model, tok, s1 + "\n" + s2, device)
            log_r = lp_s1n + lp_s2 - lp_s1s2
            ratios[ki].append(log_r)
            pool_log_rs.append(log_r)

        mean_log_r = sum(ratios[ki]) / len(ratios[ki])
        min_log_r = min(ratios[ki])
        max_log_r = max(ratios[ki])
        results[ki] = {
            "mean_log_ratio": float(mean_log_r),
            "min_log_ratio": float(min_log_r),
            "max_log_ratio": float(max_log_r),
            "mean_ratio_exp": float(math.exp(max(min(mean_log_r, 0), -50))),
        }
        print(ki, results[ki])

    return results, pool_log_rs


# -----------------------------------------------------------
# 2) 估计某个 k-shot prompt 的‘可能性’（标签条件 logprob 总和）
# -----------------------------------------------------------

def kshot_prompt_label_score(
    model,
    tok,
    task,
    examples):
    total_lp = 0.0
    total_tokens = 0
    prefix = task.instr_blind + "\n\n"
    for ex in examples:
        if len(ex) == 3:
            q, y, lab = ex
        else:
            q, y = ex
            lab = task.verbalizers[y]
        pfx = make_trec_prompt(q, label_name=lab, style=STYLE, include_label=True)
        prefix += pfx
        total_lp += text_logprob(model, tok, prefix, device)
    return total_lp, total_tokens


def estimate_c2_prompt_likelihood(
    trec,
    phi_star,
    model,
    tok,
    k_list=(1, 2, 8, 16, 64),
    R=50,
):
    results = {}
    pool = []
    for k in k_list:
        logprobs = []
        for _ in range(R):
            samp = trec.sample_k_for_task(phi_star, k_total=k)
            lp, _ = kshot_prompt_label_score(model, tok, phi_star, samp)
            logprobs.append(lp)
        pool += logprobs
        mean_lp = sum(logprobs) / len(logprobs)
        std_lp = (sum((x - mean_lp) ** 2 for x in logprobs) / len(logprobs)) ** 0.5
        results[k] = {"mean_lp": float(mean_lp), "std_lp": float(std_lp), "all": [float(v) for v in logprobs]}
        print(f"[Step 2] k={k:2d} | mean label logprob sum: {mean_lp:.2f} ± {std_lp:.2f}")
    return results, pool


# -----------------------------------------------------------
# 3) Lemma 1：随 k 增大，错误任务 φ 的相对概率 Pφ(p)/Pφ*(p) 收敛到 < ε
# -----------------------------------------------------------

def run_concentration_experiment(
    trec,
    phi_star_task,
    candidate_tasks: List,
    model,
    tok,
    k_list: Iterable[int] = (2, 4, 6, 8, 10),
    R: int = 100,
    epsilon: float = 1e-2,
    flip_prob: float = 0.0,
) -> Dict[int, float]:
    results = {}
    log_eps = math.log(epsilon)

    for k in k_list:
        success = 0
        for _ in range(R):
            samp = trec.sample_k_for_task(phi_star_task, k_total=k, flip_prob=flip_prob)
            lp_star, _ = kshot_prompt_label_score(model, tok, phi_star_task, samp)
            ok_all = True
            for phi in candidate_tasks:
                if phi.name == phi_star_task.name:
                    continue
                lp_phi, _ = kshot_prompt_label_score(model, tok, phi, samp)
                if (lp_phi - lp_star) >= log_eps:
                    ok_all = False
                    break
            if ok_all:
                success += 1
        results[k] = success / R
    return results


# -----------------------------------------------------------
# Probe：φ(context) 线性可分性（用于 Theorem 1 旁证）
# -----------------------------------------------------------

def extract_phi_context(
    model,
    tok,
    prompt: str,
    device: Optional[torch.device] = None,
    layer: int = -1,
    pool: str = "mean",  # ["mean", "last"]
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        out = model(input_ids, attention_mask=attn, output_hidden_states=True)
        hs = out.hidden_states[layer][0]  # [L, H]
        if pool == "mean":
            mask = attn[0].unsqueeze(-1)  # [L,1]
            vec = (hs * mask).sum(0) / mask.sum()
        else:
            valid_len = int(attn[0].sum().item())
            vec = hs[valid_len - 1]
        return vec.detach().cpu()


def probe_task_identification(
    trec,
    model,
    tok,
    candidate_tasks: List,
    k: int = 4,
    per_task_prompts: int = 100,
    layer: int = -1,
    pool: str = "mean",
    seed: int = 0,
) -> Dict[str, float]:
    import numpy as np
    rng = random.Random(seed)

    X, y = [], []
    for cls, task in enumerate(candidate_tasks):
        for _ in range(per_task_prompts):
            exs = trec.sample_k_for_task(task, k_total=k)
            prefix = task.instr_blind + '\n\n'
            for text, _, label in exs:
                prefix += make_trec_prompt(text, label_name=label, style=STYLE, include_label=True)
            vec = extract_phi_context(model, tok, prefix, layer=layer, pool=pool)
            X.append(vec.numpy())
            y.append(cls)
    X = np.asarray(X)
    y = np.asarray(y)

    # 划分训练/测试
    n = len(X)
    idx = list(range(n))
    rng.shuffle(idx)
    cut = int(n * 0.8)
    tr_idx, te_idx = idx[:cut], idx[cut:]
    Xtr, Ytr = X[tr_idx], y[tr_idx]
    Xte, Yte = X[te_idx], y[te_idx]

    # 线性 probe
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, n_jobs=1)
        clf.fit(Xtr, Ytr)
        acc = float(clf.score(Xte, Yte))
    except Exception:
        # 退化版
        Xtr_ = torch.from_numpy(Xtr)
        Ytr_ = torch.from_numpy(Ytr)
        C = int(y.max() + 1)
        Ytr_oh = torch.nn.functional.one_hot(Ytr_, num_classes=C).float()
        W, _ = torch.lstsq(Ytr_oh, torch.cat([Xtr_, torch.ones(len(Xtr_), 1)], dim=1))
        logits = Xte @ W[:Xtr_.shape[1]].numpy().T
        acc = float((logits.argmax(1) == Yte).mean())

    # Fisher ratio（稳健）
    try:
        classes = sorted(set(Ytr))
        means = [Xtr[Ytr == c].mean(0) for c in classes if (Ytr == c).sum() > 1]
        if len(means) < 2:
            fr = float("nan")
        else:
            overall = Xtr.mean(0)
            sb = sum(((m - overall) ** 2).sum() for m in means) / max(len(means) - 1, 1)
            sw = 0.0
            for c in classes:
                subset = Xtr[Ytr == c]
                if len(subset) > 1:
                    sw += subset.var(0).sum()
            sw = max(sw, 1e-8)
            fr = float(sb / sw)
            if not np.isfinite(fr):
                fr = float("nan")
    except Exception:
        fr = float("nan")

    return {"probe_acc": acc, "fisher_ratio": fr}


# -----------------------------------------------------------
# Theorem 1：margin 提升不等式的经验验证
#   Δ0(x;y,ŷ) = log P(y|x) - log P(ŷ|x)
#   Δk(p,x;y,ŷ) > Δ0/2 + (log c1 + log c2)
# -----------------------------------------------------------




def kshot_margin(model, tok, phi_task, question: str, y: str, y_tilde: str, kshot_examples=[], device='cuda') -> float:
    if device is None:
        device = next(model.parameters()).device
    prefix = phi_task.instr_blind + "\n\n"
    for q, labid, lab in kshot_examples:
        prefix += make_trec_prompt(q, label_name=lab, style=STYLE, include_label=True)
    prefix += make_trec_prompt(question, label_name=None, style=STYLE, include_label=False)
    lp_y, _ = label_conditional_logprob(model, tok, prefix, y, device)
    lp_yt, _ = label_conditional_logprob(model, tok, prefix, y_tilde, device)
    return lp_y - lp_yt









def run_theorem1_experiment(
    trec,
    phi_task,
    model, tok,
    k_list=(1, 2, 4, 6, 8, 10, 12, 16),
    R=50,
    max_tests=100,
    device=None,
    log_c1_pool=None,      # Assumption 2 的 log_r 样本池
    log_c2_pool=None,      # Assumption 3 的 log_r 样本池
    quant_c1=0.05,
    quant_c2=0.01
) -> Dict[str, object]:
    if device is None:
        device = next(model.parameters()).device

    log_c1_hat = float(np.quantile(np.array(log_c1_pool, dtype=np.float64), quant_c1))
    log_c2_hat = float(np.quantile(np.array(log_c2_pool, dtype=np.float64), quant_c2))
    theta = log_c1_hat + log_c2_hat

    # 准备测试样本：仅保留 Δ0>0
    ds = trec.sample_k_for_task(phi_task, k_total=max_tests * 2)
    tests = []
    for (q, yid, y_true) in ds:
        y_alt = phi_task.verbalizers[1 - yid]
        d0 = kshot_margin(model, tok, phi_task, q, y_true, y_alt, device)
        if d0 > 0:
            tests.append((q, y_true, y_alt, d0))
        if len(tests) >= max_tests:
            break

    per_k_success = {}
    for k in k_list:
        succ_cnt = 0
        tot = 0
        for (q, y_true, y_alt, d0) in tests:
            ok = 0
            for _ in range(R):
                exs = trec.sample_k_for_task(phi_task, k_total=k)
                dk = kshot_margin(model, tok, phi_task, q, y_true, y_alt, exs, device)
                if dk > (d0 / 2.0 + theta):
                    ok += 1
            succ_cnt += (ok / R) >= 0.5
            tot += 1
        per_k_success[k] = succ_cnt / max(tot, 1)
        print(f"[Theorem1] k={k:2d} | succ@per-x>=0.5 = {per_k_success[k]:.3f}  (theta={theta:.2f})")

    return {"theta": float(theta), "log_c1_hat": float(log_c1_hat), "log_c2_hat": float(log_c2_hat), "per_k_success": {int(k): float(v) for k, v in per_k_success.items()}}


def prepare_downstream_tasks(trec):
    tasks = []
    tasks.append(trec.build_binary_subtask("HUM", "LOC"))
    # tasks.append(trec.build_binary_subtask("NUM", "DESC"))
    # tasks.append(trec.build_binary_subtask("HUM", "ENTY"))
    tasks.append(trec.build_binary_subtask("HUM", "NUM"))
    tasks.append(trec.build_binary_subtask("LOC", "NUM"))
    # tasks.append(trec.build_binary_subtask("ABBR", "ENTY"))
    return tasks


import json
import math
import matplotlib.pyplot as plt
import os

def plot_from_json(json_path: str, save_dir: str = "."):
    """从保存的 results_xxx.json 文件中读取并绘制对比图。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    phi_name = data["task"]
    model_name = data["model"]
    EPSILON = data.get("epsilon", 5e-2)
    K_LIST = data.get("k_list", [])
    theorem1 = data["theorem1"]
    probes = data["probe"]
    lemma1 = data["lemma1_concentration"]

    # --- 从 JSON 重构数据 ---
    xs = sorted([int(k) for k in lemma1.keys()])
    ys_conc = [lemma1[str(k)] if isinstance(k, int) else lemma1[k] for k in map(str, xs)]

    ys_probe = []
    for i, k in enumerate(xs):
        if i < len(probes):
            ys_probe.append(probes[i]["probe_acc"])
        else:
            ys_probe.append(None)

    theo_map = theorem1.get("per_k_success", {})
    ys_theo = [theo_map.get(str(k), theo_map.get(int(k), 0.0)) for k in xs]

    # --- 绘图 ---
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(xs, ys_conc, marker="o", label="Concentration (Lemma 1)")
    ax1.plot(xs, ys_theo, marker="^", linestyle="--", label="Margin Success (Theorem 1)")
    ax1.set_xlabel("k (shots)")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(xs, ys_probe, marker="s", color="tab:red", label="Probe Acc (φ identifiability)")
    ax2.set_ylabel("Probe Accuracy")
    ax2.set_ylim(0, 1)

    # --- 图例合并 ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    # --- 标题 ---
    log_c1 = theorem1.get("log_c1_hat", None)
    log_c2 = theorem1.get("log_c2_hat", None)
    theta = theorem1.get("theta", None)
    plt.title(f"{phi_name}: Lemma1 vs Probe vs Theorem1 (ε={EPSILON})\n"
              f"log c1≈{log_c1:.1f}, log c2≈{log_c2:.1f}, θ≈{theta:.1f} | {model_name}")

    plt.tight_layout()
    save_name = os.path.join(save_dir, f"plot_{phi_name}_from_json.png")
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"[Plot] saved: {save_name}")

if __name__ == "__main__":
    trec = TrecICLDataset(split="train", seed=42)

    # 选择模型名称
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print("Using model:", model_name)
    # 加载 tokenizer 与模型
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.to(device)
    model.eval()

    tasks = prepare_downstream_tasks(trec)
    print("\nPrepared downstream tasks:", [t.name for t in tasks])

    for phi_star in tasks:
        print(f"\n=== Analyzing task φ* = {phi_star.name} ===")

        # ---- Step 1: Assumption 2 独立性 c1 估计（含 log_r 样本池）
        c1_stats_blind, c1_pool = estimate_c1_independence(
            trec, phi_star, model, tok, trials=100, blind=True
        )
        c1_stats, _ = estimate_c1_independence(
            trec, phi_star, model, tok, trials=100, blind=False
        )
        print(f"[Step 1] independence ratio stats: {c1_stats}; if blind instr: {c1_stats_blind}")

        # ---- Step 2: 估计某个 k-shot prompt 的‘可能性’（标签条件 logprob/每 token）
        lp_stats, c2_pool = estimate_c2_prompt_likelihood(trec, phi_star, model, tok, R=50)

        # ---- Step 3: Lemma 1 – Pφ(p)/Pφ*(p) < ε 的经验概率随 k 增长
        candidates = [t for t in tasks if t is not phi_star]
        curve = run_concentration_experiment(
            trec, phi_star, candidates, model, tok,
            k_list=K_LIST, R=100, epsilon=EPSILON, flip_prob=0.0
        )
        print(f"[Step 3] success probability curve (epsilon={EPSILON}):", curve)

        # ---- Probe（φ 识别能力）
        probes = []
        for k in K_LIST:
            probe = probe_task_identification(trec, model, tok, candidates, k=k)
            print(f"[Step 3, k={k}] probe acc: {probe['probe_acc']:.3f}, fisher ratio: {probe['fisher_ratio']:.3f}")
            probes.append(probe)

        # ---- Theorem 1：margin 不等式实验
        theorem1 = run_theorem1_experiment(
            trec, phi_star, model, tok,
            k_list=[k for k in K_LIST if k > 0],
            R=40, max_tests=80,
            device=device,
            log_c1_pool=c1_pool,
            log_c2_pool=c2_pool,
            quant_c1=0.05, quant_c2=0.01
        )

        

        # ---- 保存 JSON：包含 Assump2/3, Lemma1, Probe, Theorem1 全部数据
        out = {
            "task": phi_star.name,
            "model": model_name,
            "style": STYLE,
            "epsilon": EPSILON,
            "k_list": list(map(int, K_LIST)),
            "assumption2": {
                "with_instr": c1_stats,
                "blind": c1_stats_blind,
                "note": "values are stats over log-ratios; mean_ratio_exp is for display only"
            },
            "assumption3_label_lp": lp_stats,
            "lemma1_concentration": {int(k): float(v) for k, v in curve.items()},
            "probe": [{"probe_acc": float(p["probe_acc"]), "fisher_ratio": (None if math.isnan(p['fisher_ratio']) else float(p["fisher_ratio"]))} for p in probes],
            "theorem1": theorem1
        }
        json_name = f"results_{phi_star.name}.json"
        with open(json_name, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[JSON] saved: {json_name}")
        
        # ---- 绘图：Lemma1（concentration）+ Probe + Theorem1（margin succ）
        try:
            plot_from_json(f"results_{phi_star.name}.json", save_dir=".")
        except Exception as e:
            print("Plot skipped:", e)