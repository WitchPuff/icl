from dataset import TrecICLDataset
from typing import List, Dict, Iterable, Optional
import math
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
    
STYLE = "qa"  # ["instruction", "qa", "minimal"]
K_LIST= tuple(list(range(0, 5))+list(range(6, 16+1, 2)))  # k values to test
EPSILON = 5e-2  # concentration threshold
print("Using prompt style:", STYLE)
device = "cuda" if torch.cuda.is_available() else "cpu"


def to_device(model) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return device


def text_logprob(model, tok, text: str, device='cuda') -> float:
    """整段文本的对数似然 log P(text)。
    实现方式：标准语言模型损失（shift）取负号并乘以 token 数。
    """
    with torch.no_grad():
        enc = tok(text, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        outputs = model(input_ids, attention_mask=attn, labels=input_ids)
        # loss 是平均 per-token 的 CE, 取负再乘 token 数 => 总 logprob
        n_tokens = attn.sum().item()
        return -outputs.loss.item() * n_tokens


def label_conditional_logprob(
    model,
    tok,
    prefix: str,
    label_text: str,
    device: torch.device,
):
    label_text += '\n'
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
        label_ids = ids_full[:, n_pref:]
        context_ids = ids_full[:, :-1]
        next_token_logprobs = logprobs[:, :-1, :]
        gather = next_token_logprobs.gather(-1, ids_full[:, 1:].unsqueeze(-1)).squeeze(-1)  # [1, L-1]
        label_lp = gather[:, n_pref-1:].sum().item()
        n_label_tokens = label_ids.numel()
        # debug
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
    add_random_sep=True # 是否添加随机分隔符，默认 False
) -> str:
    """
    构造 TREC few-shot 示例的 prompt。
    add_random_sep=True 可打断上下文依赖，用于测试 Assumption 2。
    """
    sep = random_sep() if add_random_sep else "\n"
    prefix = ""

    if include_label and label_name is not None:
        prefix = f"{sep}### Example:{sep}"

    if style == "instruction":
        # instruction 模式，但不泄漏任务名
        prefix += f"Question: {question.strip()}\nPlease answer with the correct category."
        if include_label and label_name is not None:
            return f"{prefix} {label_name.strip().lower()}\n\n"
        else:
            return prefix + " "

    elif style == "qa":
        # 简单 Q/A 格式
        prefix += f"Q: {question.strip()}\nA:"
        if include_label and label_name is not None:
            return f"{prefix} {label_name.strip().lower()}\n\n"
        else:
            return prefix + " "

    elif style == "minimal":
        # 最精简格式
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
) -> float:
    task = trec.get_original_task()
    if 0 not in k_list: k_list = (0,) + tuple(k_list)
    correct = [0 for k in k_list]
    total = [0 for k in k_list]
    if max_eval is not None:
        original = trec.sample_k_for_task(task, k_total=max_eval, flip_prob=flip_prob)
    for i, k in tqdm(enumerate(k_list), total=len(k_list), desc="Evaluating TREC"):
        print(f"  k = {k}")
        prefix = task.instr + "\n\n"
        for (text, _lid, name) in original:
            if k > 0:
                support = trec.sample_k_for_task(task, k_total=k, flip_prob=flip_prob)
                for (sq, _sy, slab) in support:
                    prefix += make_trec_prompt(sq, label_name=slab, style=STYLE, include_label=True)
            prefix += make_trec_prompt(text, label_name=None, style=STYLE, include_label=False)
            scores = []
            for cname in trec.verbalizers.values():
                label = f"{cname.lower()}\n"
                lp, _ = label_conditional_logprob(model, tok, prefix, label, device)
                scores.append((lp, cname))
            pred = max(scores, key=lambda x: x[0])[1]
            if pred == name:
                correct[i] += 1
            total[i] += 1
    return {k: c / t if t > 0 else 0 for c, t, k in zip(correct, total, k_list)}


# -----------------------------------------------------------
# 1) 假设 2 的近似检验：独立性下界 c1 的经验估计
#    r = P(s1 + "\n") * P(s2) / P(s1 + "\n" + s2) ∈ [c1, 1]
# -----------------------------------------------------------

def estimate_c1_independence(
    trec,
    task,
    model,
    tok,
    trials=100,
    k_list=(1, 2, 8, 16, 64),
    blind=False,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    ratios = {k:[] for k in k_list if k > 0}
    results = {}
    for ki in k_list:
        for ti in range(trials):
            if ki == 0:
                continue
            # 各取一个样本块作为 s1, s2（k_each=1）
            samp1 = trec.sample_k_for_task(task, k_total=ki)
            samp2 = trec.sample_k_for_task(task, k_total=ki)
            # 构造两个块的字符串
            if blind:
                s1 = task.instr_blind + "\n\n"
            else:    
                s1 = task.instr + "\n\n"
            for ex in samp1:
                if len(ex) == 3:
                    q, y, _ = ex
                else:
                    q, y = ex
                s1 += make_trec_prompt(q, label_name=task.verbalizers[y], style=STYLE, include_label=True)
            s2 = ""
            for ex in samp2:
                if len(ex) == 3:
                    q, y, _ = ex
                else:
                    q, y = ex
                s2 += make_trec_prompt(q, task.verbalizers[y], style=STYLE, include_label=True)
            # 估计 log 概率
            lp_s1n = text_logprob(model, tok, s1 + "\n", device)
            lp_s2 = text_logprob(model, tok, s2, device)
            lp_s1s2 = text_logprob(model, tok, s1 + "\n" + s2, device)
            # r = math.exp(lp_s1n + lp_s2 - lp_s1s2)
            # ratios[ki].append(max(min(r, 1.0), 1e-12))
            log_r = lp_s1n + lp_s2 - lp_s1s2
            ratios[ki].append(log_r)
        mean_log_r = sum(ratios[ki]) / len(ratios[ki])
        min_log_r = min(ratios[ki])
        max_log_r = max(ratios[ki])
        # 如需可视化，可再 exp 回去（限制范围防止上溢）
        results[ki] = {
            "mean_log_ratio": mean_log_r,
            "min_log_ratio": min_log_r,
            "max_log_ratio": max_log_r,
            "mean_ratio_exp": math.exp(max(min(mean_log_r, 0), -50)),  # 仅展示用
        }
        print(ki, results[ki])
        
    return results


# -----------------------------------------------------------
# 2) 估计某个 k-shot prompt 被模型产生的“可能性”
#    （这里用标签条件 logprob 总和作为 score）
# -----------------------------------------------------------

def kshot_prompt_label_score(
    model,
    tok,
    task,
    examples):
    total_lp = 0.0
    total_tokens = 0
    lab = None
    prefix = task.instr_blind + "\n\n"
    for ex in examples:
        if len(ex) == 3:
            q, y, lab = ex
        else:
            q, y = ex
        pfx = make_trec_prompt(q, label_name=lab, style=STYLE, include_label=True)
        prefix += pfx
        total_lp += text_logprob(model, tok, prefix, device)
        # lp, n = label_conditional_logprob(model, tok, prefix, lab, device)
        # total_lp += lp
        # total_tokens += n
    return total_lp, total_tokens




def prepare_downstream_tasks(trec):
    tasks = []
    # tasks.append(trec.get_original_task())  # 原始 6 类任务
    tasks.append(trec.build_binary_subtask("HUM", "LOC"))
    # tasks.append(trec.build_binary_subtask("HUM", "NUM"))
    # tasks.append(trec.build_binary_subtask("LOC", "NUM"))
    tasks.append(trec.build_binary_subtask("DESC", "NUM"))
    tasks.append(trec.build_binary_subtask("ABBR", "ENTY"))
    return tasks


# -----------------------------------------------------------
# 3) 关键测量：随 k 增大，错误任务 φ 的相对概率 Pφ(p)/Pφ*(p) 收敛到 < ε。
#    我们用标签条件 logprob 代替 log P(p)，比较 Δ = log Pφ(p) - log Pφ*(p)。
#    要求：Δ < log ε 。重复 R 次抽样，统计满足比例，画随 k 的曲线。
# -----------------------------------------------------------

def extract_phi_context(
    model,
    tok,
    prompt: str,
    device: Optional[torch.device] = None,
    layer: int = -1,
    pool: str = "mean",  # ["mean", "last"]
) -> torch.Tensor:
    """提取 φ(context) 向量：把 prompt 过一遍模型，取指定层的 hidden states 聚合。
    - mean: 对所有 token 表示求平均（常见做法）
    - last: 取最后一个 token 的表示
    返回 shape: [hidden_size]
    """
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
    """用线性 probe 识别 φ（任务）——验证 Theorem 1/φ(context) 思路。
    过程：
      1) 对每个候选任务 φ，重复 per_task_prompts 次：均匀采样 k-shot 例子，拼成 prompt；
      2) 提取 φ(context) 表示（某层 hidden states 的 mean/last）
      3) 训练一个线性分类器（LogisticRegression），做 80/20 划分评估
    返回：{"probe_acc": ..., "fisher_ratio": ...}
    """
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

    # 线性 probe（sklearn 优先，若无则用最小二乘）
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, n_jobs=1)
        clf.fit(Xtr, Ytr)
        acc = float(clf.score(Xte, Yte))
        W = clf.coef_  # [C, H]
    except Exception:
        # 退化版：一对多最小二乘
        Xtr_ = torch.from_numpy(Xtr)
        Ytr_ = torch.from_numpy(Ytr)
        C = int(y.max() + 1)
        Ytr_oh = torch.nn.functional.one_hot(Ytr_, num_classes=C).float()
        W, _ = torch.lstsq(Ytr_oh, torch.cat([Xtr_, torch.ones(len(Xtr_), 1)], dim=1))  # 简化
        logits = Xte @ W[:Xtr_.shape[1]].numpy().T
        acc = float((logits.argmax(1) == Yte).mean())

    # Fisher ratio（类间均值方差 / 类内方差）——粗略版
    # Fisher ratio（类间方差 / 类内方差）
    try:
        import numpy as np
        classes = sorted(set(y))
        means = [X[Ytr == c].mean(0) for c in classes if (Ytr == c).sum() > 1]
        if len(means) < 2:
            fr = float("nan")
        else:
            overall = Xtr.mean(0)
            # between-class variance
            sb = sum(((m - overall) ** 2).sum() for m in means) / max(len(means) - 1, 1)
            # within-class variance
            sw = 0.0
            for c in classes:
                subset = Xtr[Ytr == c]
                if len(subset) > 1:
                    sw += subset.var(0).sum()
            sw = max(sw, 1e-8)  # 防除零
            fr = float(sb / sw)
            if not np.isfinite(fr):
                fr = float("nan")
    except Exception:
        fr = float("nan")

    return {"probe_acc": acc, "fisher_ratio": fr}

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
            # 从 φ* 任务均匀采样 k 条 few-shot
            samp = trec.sample_k_for_task(phi_star_task, k_total=k, flip_prob=flip_prob)

            # φ* 的分数（标签条件 logprob 总和）
            lp_star, _ = kshot_prompt_label_score(model, tok, phi_star_task, samp)

            # 检查所有其他 φ 的 Δ 是否都 < log ε
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

def estimate_c2_prompt_likelihodd(
    trec,
    phi_star,
    model,
    tok,
    k_list=(1, 2, 8, 16, 64),
    R=50,
):
    """
    对不同的 few-shot k，重复 R 次采样，估计每个 prompt 的标签条件 logprob。
    返回一个 {k: [logprob_sum_1, ..., logprob_sum_R]} 字典。
    """

    results = {}

    for k in k_list:
        logprobs = []
        for _ in range(R):
            # 均匀采样 k 条样例（可以加入 flip_prob 如有）
            samp = trec.sample_k_for_task(phi_star, k_total=k)
            lp, _ = kshot_prompt_label_score(model, tok, phi_star, samp)
            logprobs.append(lp)
                
        mean_lp = sum(logprobs) / len(logprobs)
        std_lp = (sum((x - mean_lp) ** 2 for x in logprobs) / len(logprobs)) ** 0.5
        results[k] = {
            "mean_lp": mean_lp,
            "std_lp": std_lp,
            "all": logprobs,
        }
        print(f"[Step 2] k={k:2d} | mean label logprob sum: {mean_lp:.2f} ± {std_lp:.2f}")

    return results




if __name__ == "__main__":
    trec = TrecICLDataset(split="train", seed=42)


    # 选择模型名称
    # model_name = "EleutherAI/pythia-1b"
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

    # ---- Step 0: 原始 6 类零样例准确率（子集）
    # acc = test_trec(trec, model, tok, max_eval=100, k_list=K_LIST, flip_prob=0)
    # print(f"[Step 0] Zero-shot 6-class accuracy (subset): {acc:.3f}")

    tasks = prepare_downstream_tasks(trec)
    print("\nPrepared downstream tasks:", [t.name for t in tasks])
    for phi_star in tasks:
        print(f"\n=== Analyzing task φ* = {phi_star.name} ===")
        # ---- Step 1: Assumption 2 独立性 c1 估计
        c1_stats_blind = estimate_c1_independence(trec, phi_star, model, tok, trials=100, blind=True)
        c1_stats = estimate_c1_independence(trec, phi_star, model, tok, trials=100, blind=False)
        print(f"[Step 1] independence ratio stats: {c1_stats}; if blind instr: {c1_stats_blind}")

        # ---- Step 2: 估计某个 k-shot prompt 的‘可能性’（标签条件 logprob/每 token）
        estimate_c2_prompt_likelihodd(trec, phi_star, model, tok, R=50)

        # ---- Step 3: 关键曲线：Pφ(p)/Pφ*(p) < ε 的经验概率随 k 增长
        candidates = [t for t in tasks if t is not phi_star]  # 以 HUM/LOC/NUM 的三对任务作为竞争者
        curve = run_concentration_experiment(
            trec, phi_star, candidates, model, tok,
            k_list=K_LIST, R=100, epsilon=EPSILON, flip_prob=0.0
        )
        print(f"[Step 3] success probability curve (epsilon={EPSILON}):", curve)
        probes = []
        for k in K_LIST:
            probe = probe_task_identification(trec, model, tok, candidates, k=k)
            print(f"[Step 3, k={k}] probe acc: {probe['probe_acc']:.3f}, fisher ratio: {probe['fisher_ratio']:.3f}")
            probes.append(probe)
        # 如需画图：
        try:
            import matplotlib.pyplot as plt
            # --- 数据准备 ---
            xs = sorted(curve.keys())                           # concentration 的横轴
            ys_conc = [curve[k] for k in xs]                    # concentration 成功概率
            ys_probe = [probes[i]["probe_acc"] for i, k in enumerate(xs)]  # probe 准确率

            fig, ax1 = plt.subplots(figsize=(6,4))

            # 左轴：concentration
            color1 = "tab:blue"
            ax1.set_xlabel("k (shots)")
            ax1.set_ylabel("Concentration  Pφ/Pφ* < ε", color=color1)
            ax1.plot(xs, ys_conc, marker="o", color=color1, label="Concentration (Lemma 1)")
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis="y", labelcolor=color1)
            ax1.grid(True, linestyle="--", alpha=0.5)

            # 右轴：probe accuracy
            ax2 = ax1.twinx()
            color2 = "tab:red"
            ax2.set_ylabel("Probe Accuracy", color=color2)
            ax2.set_ylim(0, 1)
            ax2.plot(xs, ys_probe, marker="s", color=color2, label="Probe Acc (Theorem 1)")
            ax2.tick_params(axis="y", labelcolor=color2)

            # 标题与图例
            plt.title("Concentration vs Task Identifiability across k shots")
            fig.tight_layout()
            fig.savefig("lemma1_probe_vs_concentration.png", dpi=300)
                        

        except Exception as e:
            print("Plot skipped:", e)

