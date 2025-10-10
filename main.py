from dataset import TrecICLDataset
from typing import List, Dict, Optional
import math
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import wandb
import json, os
import matplotlib.pyplot as plt
import numpy as np

STYLE = "qa"  # ["instruction", "qa", "minimal"]
K_LIST = tuple(list(range(0, 5)) + list(range(6, 16, 4)))  # k values to test
FLIP = 0
EPSILON = 1e-2  # concentration threshold
DELTA = 1e-2  # margin success threshold
device = "cuda" if torch.cuda.is_available() else "cpu"
add_random_sep = True
print("Using prompt style:", STYLE)
print("Using device:", device)
print("Using random separator in prompts:", add_random_sep)
config = {
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    "style": STYLE,
    "k_list": K_LIST,
    "epsilon": EPSILON,
    "add_random_sep": add_random_sep
}
wandb.init(project="icl", name=f"{STYLE}_{FLIP}_{EPSILON}_{DELTA}", config=config)

def to_device(model) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return device


def text_logprob(model, tok, text: str, device='cuda') -> float:
    with torch.no_grad():
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=model.config.max_position_embeddings,
        )
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
    tag = "".join(random.choices(string.ascii_uppercase, k=length))
    return f"\n### SEP_{tag} ###\n"


def make_trec_prompt(
    question="",
    label_name=None,
    style="qa",          # ["instruction", "qa", "minimal"]
    include_label=True,
    add_random_sep=add_random_sep 
) -> str:
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





def estimate_c1_independence(
    trec,
    task,
    model,
    tok,
    k_list,
    trials=100,
    blind=False,
) -> Dict[int, Dict[str, float]]:
    device = next(model.parameters()).device
    ratios = {k: [] for k in k_list if k > 0}
    ratios_per_token = {k: [] for k in k_list if k > 0}
    pool_log_rs: List[float] = []

    print(f"[Step 1] Estimating independence ratios for task {task.name}, blind={blind}...")
    results = {}

    for ki in k_list:
        if ki == 0:
            continue
        for _ in range(trials):
            samp1 = trec.sample_k_for_task(task, k_total=ki, flip_prob=FLIP)
            samp2 = trec.sample_k_for_task(task, k_total=ki, flip_prob=FLIP)
            s1 = (task.instr_blind if blind else task.instr) + "\n\n"
            for ex in samp1:
                q, y = ex[0], ex[1]
                lab = ex[2] if len(ex) == 3 else task.verbalizers[y]
                s1 += make_trec_prompt(q, label_name=lab, style=STYLE, include_label=True)
            s2 = ""
            for ex in samp2:
                q, y = ex[0], ex[1]
                lab = ex[2] if len(ex) == 3 else task.verbalizers[y]
                s2 += make_trec_prompt(q, label_name=lab, style=STYLE, include_label=True)

            lp_s1n = text_logprob(model, tok, s1 + "\n", device)
            lp_s2 = text_logprob(model, tok, s2, device)
            lp_s1s2 = text_logprob(model, tok, s1 + "\n" + s2, device)

            def tok_len(text):
                ids = tok(text, truncation=True, max_length=4096)["input_ids"]
                return len(ids) if isinstance(ids[0], int) else len(ids[0])

            len1, len2, len12 = tok_len(s1 + "\n"), tok_len(s2), tok_len(s1 + "\n" + s2)

            log_r = lp_s1n + lp_s2 - lp_s1s2
            log_r_norm = (lp_s1n / len1) + (lp_s2 / len2) - (lp_s1s2 / len12)
            pool_log_rs.append(log_r_norm)
            ratios[ki].append(log_r)
            ratios_per_token[ki].append(log_r_norm)

        mean_log_r = sum(ratios[ki]) / len(ratios[ki])
        mean_log_r_token = sum(ratios_per_token[ki]) / len(ratios_per_token[ki])
        results[ki] = {
            "mean_log_ratio": float(mean_log_r),
            "mean_log_ratio_per_token": float(mean_log_r_token),
            "min_log_ratio": float(min(ratios[ki])),
            "max_log_ratio": float(max(ratios[ki])),
            "mean_ratio_exp": float(math.exp(max(min(mean_log_r, 0), -50))),
        }
        print(f"k={ki:<3d} mean_log_r={mean_log_r:8.2f}  per_token={mean_log_r_token:8.4f}")
        wandb.log({
            "k": int(ki),
            f"c1/{task.name}":
                {
                    "mean": float(mean_log_r),
                    "per_token": float(mean_log_r_token),
                    "min": float(min(ratios[ki])),
                    "max": float(max(ratios[ki])),
                    "mean_exp": float(math.exp(max(min(mean_log_r, 0), -50))),
                }
        })
    if 1 in ratios:
        mean_log_c1_fixed = sum(ratios[1]) / len(ratios[1])
    else:
        mean_log_c1_fixed = sum(pool_log_rs) / len(pool_log_rs)
    mean_log_c1_token = sum(sum(v) for v in ratios_per_token.values()) / sum(
        len(v) for v in ratios_per_token.values()
    )

    global_stats = {
        "mean_log_c1_fixed_k1": float(mean_log_c1_fixed),
        "mean_c1_fixed_k1_exp": float(math.exp(max(min(mean_log_c1_fixed, 0), -50))),
        "mean_log_c1_per_token": float(mean_log_c1_token),
        "mean_c1_per_token_exp": float(math.exp(max(min(mean_log_c1_token, 0), -50))),
    }

    print(f"[c1 summary] log_c1 ≈ {mean_log_c1_fixed:.2f}, log_c1/token ≈ {mean_log_c1_token:.4f}")
    wandb.log({
        f"c1/{task.name}/global": global_stats
    })
    return {"per_k": results, "global": global_stats, "pool": pool_log_rs}

def kshot_prompt_label_score(
    model,
    tok,
    task,
    examples):
    total_lp = 0.0
    total_tokens = 0
    prefix = task.instr_blind + "\n\n"
    for i, ex in enumerate(examples):
        if len(ex) == 3:
            q, y, lab = ex
        else:
            q, y = ex
            lab = task.verbalizers[y]
        if i == len(examples) - 1:
            lab = None
        pfx = make_trec_prompt(q, label_name=lab, style=STYLE, include_label=True)
        prefix += pfx
        total_lp += text_logprob(model, tok, prefix, device)
    return total_lp, total_tokens, prefix




def estimate_c2_prompt_likelihood(
    trec,
    phi_star,
    model,
    tok,
    k_list,
    R=50,
):

    results = {}
    pool_raw = []
    pool_per_token = []
    if 0 not in k_list:
        k_list = (0,) + tuple(k_list)
    for k in k_list:
        logprobs, logprobs_per_token = [], []
        for _ in range(R):
            samp = trec.sample_k_for_task(phi_star, k_total=k+1, flip_prob=FLIP)
            lp, _, prefix = kshot_prompt_label_score(model, tok, phi_star, samp)
            n_tokens = len(tok(prefix, truncation=True, max_length=4096).input_ids)
            lp_per_token = lp / max(n_tokens, 1)

            logprobs.append(lp)
            logprobs_per_token.append(lp_per_token)

        pool_raw += logprobs
        pool_per_token += logprobs_per_token

        mean_lp = sum(logprobs) / len(logprobs)
        std_lp = (sum((x - mean_lp) ** 2 for x in logprobs) / len(logprobs)) ** 0.5
        mean_lp_token = sum(logprobs_per_token) / len(logprobs_per_token)
        std_lp_token = (sum((x - mean_lp_token) ** 2 for x in logprobs_per_token) / len(logprobs_per_token)) ** 0.5

        results[k] = {
            "mean_lp": float(mean_lp),
            "std_lp": float(std_lp),
            "mean_lp_per_token": float(mean_lp_token),
            "std_lp_per_token": float(std_lp_token),
            "all": [float(v) for v in logprobs],
            "all_per_token": [float(v) for v in logprobs_per_token],
        }

        print(f"[Step 2] k={k:2d} | mean logP(p): {mean_lp:9.2f} ± {std_lp:6.2f} "
              f"| per-token: {mean_lp_token:8.4f} ± {std_lp_token:6.4f}")
        wandb.log({
            "k": int(k),
            f"c2/{phi_star.name}":
                {
                    "mean_log_p": float(mean_lp),
                    "std_log_p": float(std_lp),
                    "mean_log_p_per_token": float(mean_lp_token),
                    "std_log_p_per_token": float(std_lp_token),
                }
        })
    mean_lp_global = sum(pool_raw) / len(pool_raw)
    mean_lp_per_token_global = sum(pool_per_token) / len(pool_per_token)
    global_stats = {
        "mean_log_c2": float(mean_lp_global),
        "mean_log_c2_per_token": float(mean_lp_per_token_global),
        "mean_c2_per_token_exp": float(math.exp(mean_lp_per_token_global)),
    }
    print(f"[c2 summary] log_c2/token ≈ {mean_lp_per_token_global:.4f}")

    return {"per_k": results, "global": global_stats, "pool": pool_raw}


def run_concentration_experiment(
    trec,
    phi_star_task,
    candidate_tasks: List,
    model,
    tok,
    k_list,
    R: int = 100,
    epsilon: float = 1e-2
) -> Dict[int, float]:
    results = {}
    log_eps = math.log(epsilon)
    if 0 not in k_list:
        k_list = (0,) + tuple(k_list)
    
    kl_stats = [] 

    for k in k_list:
        success = 0
        kl_vals = []
        for _ in range(R):
            samp = trec.sample_k_for_task(phi_star_task, k_total=k+1, flip_prob=FLIP)
            lp_star, _, _ = kshot_prompt_label_score(model, tok, phi_star_task, samp)
            ok_all = True
            for phi in candidate_tasks:
                if phi.name == phi_star_task.name:
                    continue
                lp_phi, _, _ = kshot_prompt_label_score(model, tok, phi, samp)

                # ---- KL divergence term ----
                kl_val = lp_star - lp_phi
                kl_vals.append(kl_val)

                # ---- Lemma 1 success condition ----
                if (lp_phi - lp_star) >= log_eps:
                    ok_all = False
                    break
            if ok_all:
                success += 1
        prob = success / R
        results[k] = {"concentration_prob": prob}
        if kl_vals:
            mean_kl = sum(kl_vals) / len(kl_vals)
            results[k]["kl_mean"] = mean_kl
            results[k]["kl_min"] = min(kl_vals)
            results[k]["kl_max"] = max(kl_vals)
            kl_stats.append(mean_kl)

        wandb.log({
            "k": int(k),
            f"lemma1_concentration/{phi_star_task.name}": {
                "concentration_prob": float(prob),
                "mean_KL": float(mean_kl)
            }
        })
        print(f"[Step 2] k={k:2d} | concentration_prob: {prob:.3f} | mean_KL: {mean_kl:.2f}")

    return results



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
        enc = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=model.config.max_position_embeddings,  # 4096
        )
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
    k_list,
    per_task_prompts: int = 100,
    layer: int = -1,
    pool: str = "mean",
    seed: int = 0,
) -> Dict[str, float]:
    import numpy as np
    rng = random.Random(seed)

    if 0 not in k_list:   
        k_list = (0,) + tuple(k_list)
        
    ret = {
        k: {} for k in k_list
    }
    for k in k_list:
        X, y = [], []
        for cls, task in enumerate(candidate_tasks):
            for _ in range(per_task_prompts):
                exs = trec.sample_k_for_task(task, k_total=k+1, flip_prob=FLIP)
                prefix = task.instr_blind + '\n\n'
                for i, (text, _, label) in enumerate(exs):
                    if i == len(exs) - 1:
                        label = None
                    prefix += make_trec_prompt(text, label_name=label, style=STYLE, include_label=True)
                vec = extract_phi_context(model, tok, prefix, layer=layer, pool=pool)
                X.append(vec.numpy())
                y.append(cls)
        X = np.asarray(X)
        y = np.asarray(y)

        n = len(X)
        idx = list(range(n))
        rng.shuffle(idx)
        cut = int(n * 0.8)
        tr_idx, te_idx = idx[:cut], idx[cut:]
        Xtr, Ytr = X[tr_idx], y[tr_idx]
        Xte, Yte = X[te_idx], y[te_idx]

        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=2000, solver="lbfgs", penalty="l2", C=0.1, n_jobs=1)
            clf.fit(Xtr, Ytr)
            acc = float(clf.score(Xte, Yte))

        except Exception as e:
            print(e)
            acc = float("nan")

            

        
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
            
            
        wandb.log({
            "k": int(k),
            f"probe": {
                "probe_acc": float(acc),
            }
        })
        ret[k]["probe_acc"] = acc
        print(f"[Step 0] k={k} probe_acc: {acc:.4f}")
        wandb.log({
            "k": int(k),
            f"probe": {
                "fisher_ratio": float(fr),
            }
        })
        ret[k]["fisher_ratio"] = fr
        print(f"[Step 0] k={k} fisher_ratio: {fr:.4f}")
    return ret




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
    model, 
    tok,
    c1_stats,
    c2_stats,
    k_list,
    R=50,
    max_tests=100,
    delta=1e-2,
    device=None,
) -> Dict[str, object]:
    if device is None:
        device = next(model.parameters()).device

    log_c1_per_tok = c1_stats["global"]["mean_log_c1_per_token"]
    log_c2_per_tok = c2_stats["global"]["mean_log_c2_per_token"]

    # Δ0>0 selection
    ds = trec.sample_k_for_task(phi_task, k_total=max_tests * 2, flip_prob=FLIP)
    tests = []
    for (q, yid, y_true) in ds:
        y_alt = phi_task.verbalizers[1 - yid]
        d0 = kshot_margin(model, tok, phi_task, q, y_true, y_alt, device=device)
        if d0 > 0:
            tests.append((q, y_true, y_alt, d0))
        if len(tests) >= max_tests:
            break
    print(f"[Step 2] selected {len(tests)} tests with Δ0 > 0")

    per_k_success = {}
    if 0 not in k_list:
        k_list = (0,) + tuple(k_list)

    for k in k_list:
        succ_cnt = 0
        tot = 0

        for (q, y_true, y_alt, d0) in tests:
            ok = 0
            for _ in range(R):
                # sample k-shot examples
                exs = trec.sample_k_for_task(phi_task, k_total=k, flip_prob=FLIP)

                # construct full prompt and compute token length
                prefix = phi_task.instr_blind + '\n\n'
                for i, (text, _, label) in enumerate(exs):
                    if i == len(exs) - 1:
                        label = None
                    prefix += make_trec_prompt(text, label_name=label, style=STYLE, include_label=True)

                # count total tokens in this prompt
                with torch.no_grad():
                    L = len(tok(prefix, truncation=True, max_length=4096)["input_ids"])

                # total-scale constants for this prompt
                theta = L * 2 * log_c1_per_tok

                # compute contextual margin
                dk = kshot_margin(model, tok, phi_task, q, y_true, y_alt, exs, device)

                if dk > (d0 / 2.0 + theta):
                    ok += 1

            succ_cnt += (ok / R) >= 1-delta
            tot += 1

        per_k_success[k] = succ_cnt / max(tot, 1)
        print(f"[Theorem1] k={k:2d} | succ@per-x>={1-delta} = {per_k_success[k]:.3f} (mean L≈{L}, θ={theta:.2f})")

        wandb.log({
            "k": int(k),
            f"theorem1/{phi_task.name}": {
                "success_prob": float(per_k_success[k]),
                "mean_L": float(L),
                "theta": float(theta)
            }
        })

    wandb.log({
        f"theorem1/{phi_task.name}": {
            "log_c1_hat": float(log_c1_per_tok),
            "log_c2_hat": float(log_c2_per_tok),
        }
    })

    return {
        "log_c1_hat": float(log_c1_per_tok),
        "log_c2_hat": float(log_c2_per_tok),
        "per_k_success": {int(k): float(v) for k, v in per_k_success.items()},
    }


def prepare_downstream_tasks(trec):
    tasks = []
    tasks.append(trec.build_binary_subtask("HUM", "LOC"))
    tasks.append(trec.build_binary_subtask("NUM", "DESC"))
    tasks.append(trec.build_binary_subtask("HUM", "ENTY"))
    tasks.append(trec.build_binary_subtask("HUM", "NUM"))
    tasks.append(trec.build_binary_subtask("LOC", "NUM"))
    tasks.append(trec.build_binary_subtask("ABBR", "ENTY"))
    return tasks



    
if __name__ == "__main__":
    trec = TrecICLDataset(split="train", seed=42)

    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print("Using model:", model_name)
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
    
    probes = probe_task_identification(trec, model, tok, tasks, k_list=K_LIST)
    print(probes)
    
    for phi_star in tasks:
        print(f"\n=== Analyzing task φ* = {phi_star.name} ===")

        c1_stats_blind = estimate_c1_independence(
            trec, phi_star, model, tok, k_list=[k for k in K_LIST if k > 0], trials=50, blind=True
        )
        # c1_stats = estimate_c1_independence(
        #     trec, phi_star, model, tok, trials=100, blind=False
        # )
        print(f"[Step 1] independence ratio stats if blind instr: {c1_stats_blind}")

        c2_stats = estimate_c2_prompt_likelihood(trec, phi_star, model, tok, k_list=K_LIST, R=50)

        candidates = [t for t in tasks if t is not phi_star]
        curve = run_concentration_experiment(
            trec, phi_star, candidates, model, tok,
            k_list=K_LIST, R=100, epsilon=EPSILON
        )
        print(f"[Step 3] success probability curve (epsilon={EPSILON}):", curve)



        theorem1 = run_theorem1_experiment(
            trec, phi_star, model, tok, c1_stats_blind, c2_stats,
            k_list=[k for k in K_LIST if k > 0],
            R=50, max_tests=80, delta=DELTA,
            device=device
        )

        

        out = {
            "task": phi_star.name,
            "model": model_name,
            "style": STYLE,
            "epsilon": EPSILON,
            "delta": DELTA,
            "k_list": list(map(int, K_LIST)),
            "assumption2": {
                # "with_instr": c1_stats,
                "blind": c1_stats_blind,
            },
            "assumption3_label_lp": c2_stats,
            "lemma1_concentration": curve,
            "probe": probes,
            "theorem1": theorem1
        }
        json_name = f"results_{phi_star.name}.json"
        with open(json_name, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[JSON] saved: {json_name}")
        