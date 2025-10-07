import os
import math
import argparse
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import matplotlib.pyplot as plt
import math


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x

@dataclass
class Task:
    name: str
    verbalizers: Dict[int, str]  # label id -> text verbalization
    data: Dict[str, str]

# TREC coarse labels mapping (6 classes)
TREC_LABELS = ["DESC", "ENTY", "ABBR", "HUM", "LOC", "NUM"]
TREC_VERBALIZERS = {
    0: "description",   # DESC
    1: "entity",        # ENTY
    2: "abbreviation",  # ABBR
    3: "human",         # HUM
    4: "location",      # LOC
    5: "number",        # NUM
}

def build_perm_task(seed=0):

    rng = np.random.default_rng(seed)
    ids = list(TREC_VERBALIZERS.keys())

    perm = ids.copy()
    rng.shuffle(perm)
    mapping = {lbl: TREC_VERBALIZERS[perm[i]] for i, lbl in enumerate(ids)}
    name = f"trec_perm_seed{seed}"

    return Task(name=name, verbalizers=mapping)

def flip_label(lbl, n_labels, rng):
    """Flip label to a random other label."""
    choices = [i for i in range(n_labels) if i != lbl]
    return int(rng.choice(choices))



class LikelihoodScorer:
    def __init__(self, model_name="gpt2", device=None):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def bpe(self, s):
        return self.tokenizer.encode(s, add_special_tokens=False)

    @torch.no_grad()
    def seq_logprob(self, prefix_ids, target_ids):
        """
        =log P(target_ids | prefix_ids)
        """
        if len(target_ids) == 0:
            return 0.0
        # GPT-2 max position embeddings 1024
        full = prefix_ids + target_ids
        if len(full) >= self.model.config.n_positions:
            # truncate from the left to fit context window
            overflow = len(full) - self.model.config.n_positions + 1
            prefix_len = max(0, len(prefix_ids) - overflow)
            prefix_ids = prefix_ids[-prefix_len:]
            full = prefix_ids + target_ids

        input_ids = torch.tensor([full], dtype=torch.long, device=self.device)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits  # [1, L, V]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Positions corresponding to target tokens are last len(target_ids) positions
        L = full.__len__()
        T = len(target_ids)
        # For autoregressive LM, prob of token at pos t uses logits at pos t-1
        # So we sum log_probs[0, (L-T-1):(L-1), target_ids]
        start = L - T - 1
        if start < 0: start = 0
        idxs = range(start, L - 1)
        target = torch.tensor(target_ids[: (L - 1 - start)], device=self.device)
        selected = log_probs[0, list(idxs), target]
        return float(selected.sum().item())



def make_prompt_example(task, text, label_id):
    cur = f"Task: {task.name}\nInput: {text}\nLabel:"
    lab = task.verbalizers[label_id].lower()
    return cur, lab

def get_loglikelihood(scorer, fewshot_batch, task):
    """
    Compute cumulative log-likelihood of labels under a task's verbalizers
    """
    total_lp = 0.0
    prefix_ids = []
    for ex in fewshot_batch:
        cur, lab = make_prompt_example(task, ex["text"], ex["label"])
        cur_prefix = scorer.bpe(cur)
        lab_ids = scorer.bpe(lab)
        total_lp += scorer.seq_logprob(prefix_ids + cur_prefix, lab_ids)
        full_ids = cur_prefix + lab_ids + scorer.bpe("\n\n")
        prefix_ids += full_ids
    return total_lp

def get_tasks_kl(scorer, dataset, task_true, task_perm, max_examples=200):
    """
    Estimate symmetric KL between predictive distributions over the label-token set
    under two tasks. We restrict the support to the first-token of each verbalizer.
    """
    # Build label-token set (first token of each verbalizer under tokenizer)
    label_texts_true = [task_true.verbalizers[i].lower() for i in range(len(TREC_LABELS))]
    label_texts_perm = [task_perm.verbalizers[i].lower() for i in range(len(TREC_LABELS))]
    # Use union of first tokens to define support
    ids_true = [scorer.bpe(s)[0] for s in label_texts_true]
    ids_perm = [scorer.bpe(s)[0] for s in label_texts_perm]
    support = sorted(list(set(ids_true + ids_perm)))

    def next_token_dist(prefix_ids):
        with torch.no_grad():
            # Truncate prefix to fit
            if len(prefix_ids) >= scorer.model.config.n_positions:
                prefix_ids = prefix_ids[-(scorer.model.config.n_positions - 1):]
            inp = torch.tensor([prefix_ids], device=scorer.device)
            logits = scorer.model(inp).logits  # [1, L, V]
            last = logits[0, -1, :]  # distribution for next token
            logp = torch.log_softmax(last, dim=-1)
            probs = torch.exp(logp)[support]
            probs = probs / probs.sum()  # renormalize on support
            return probs.detach().cpu().numpy()

    def kl(p, q, eps=1e-12):
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        p = p / p.sum(); q = q / q.sum()
        return float(np.sum(p * (np.log(p) - np.log(q))))

    # Sample examples
    rng = np.random.default_rng(0)
    sel = rng.choice(len(dataset), size=min(max_examples, len(dataset)), replace=False)
    kls = []
    for idx in sel:
        ex = dataset[int(idx)]
        # Build prefix up to "Label:" for the SAME (text, label), under two task names/verbalizers
        cur_true, _ = make_prompt_example(task_true, ex["text"], ex["label"])
        cur_perm, _ = make_prompt_example(task_perm, ex["text"], ex["label"])
        p_true = next_token_dist(scorer.bpe(cur_true))
        p_perm = next_token_dist(scorer.bpe(cur_perm))
        # Symmetric KL
        kls.append(0.5 * (kl(p_true, p_perm) + kl(p_perm, p_true)))
    return float(np.mean(kls))

def sample_examples(pool, k, rng, balanced=False):

    if not balanced:
        idxs = rng.choice(len(pool), size=k, replace=False)
        return [pool[i] for i in idxs]

    from collections import defaultdict
    label2idx = defaultdict(list)
    for i, ex in enumerate(pool):
        label2idx[ex["label"]].append(i)

    labels = list(label2idx.keys())
    n_classes = len(labels)
    per_class = max(1, k // n_classes)

    chosen = []
    for lbl in labels:
        idxs = rng.choice(label2idx[lbl], size=per_class, replace=False)
        chosen.extend(idxs)

    if len(chosen) < k:
        rest = list(set(range(len(pool))) - set(chosen))
        extra = rng.choice(rest, size=k - len(chosen), replace=False)
        chosen.extend(extra)

    rng.shuffle(chosen)
    return [pool[i] for i in chosen]

def loglikelihood_ratio_success(scorer,
                                pool,
                                task_true,
                                task_perm,
                                k,
                                trials,
                                eps,
                                flip_labels=False,
                                balanced=False,
                                seed=0):
    """
    Monte Carlo estimate of P[ P_phi(p) / P_phi*(p) < eps ] over random k-shot batches.
    Returns (success_prob, avg_ll_gap), where ll_gap = E[ LL_true - LL_perm ].
    """
    rng = np.random.default_rng(seed)
    successes = 0
    gaps = []
    n_labels = len(TREC_LABELS)

    for t in range(trials):
        # idxs = rng.choice(len(pool), size=k, replace=False)
        
        # few = []
        # for i in idxs:
        #     ex = dict(pool[int(i)])
        #     if flip_labels:
        #         ex["label"] = flip_label(ex["label"], n_labels, rng)
        #     few.append(ex)
        few = sample_examples(pool, k, rng, balanced=balanced)
        for ex in few:
            if flip_labels:
                ex["label"] = flip_label(ex["label"], n_labels, rng)
                
        ll_true = get_loglikelihood(scorer, few, task_true)
        ll_perm = get_loglikelihood(scorer, few, task_perm)
        # P_perm / P_true = exp(ll_perm - ll_true) < eps  <=>  ll_true - ll_perm > -log eps
        gap = ll_true - ll_perm
        gaps.append(gap)
        if gap > -math.log(eps):
            successes += 1

    return successes / trials, float(np.mean(gaps))







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--flip", action="store_true", help="Randomly flip labels in the few-shot prompt (robustness check in lemma 1).")
    parser.add_argument("--balanced", action="store_true", help="Randomly flip labels in the few-shot prompt (robustness check in lemma 1).")
    parser.add_argument("--max_kl_examples", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)

    print("Loading TREC (coarse) ...")
    ds = load_dataset("trec", "coarse")
    data = list(ds[args.split])
    print("Sample Count:", len(data))
    for ex in data:
        if "label" not in ex:
            ex["label"] = int(ex.get("coarse_label", ex.get("label-coarse", ex.get("label-coarse-fine", 0))))
        ex["text"] = ex["text"].strip()


    scorer = LikelihoodScorer(model_name=args.model)

    ks = args.ks
    succ_mat = []
    gap_mat = []
    kl_list = []
    threshold = -math.log(args.eps)
    print(f"Threshold (−log ε) = {threshold:.2f}")
    for r in range(args.repeat):
        task_true = Task(name="trec", verbalizers=TREC_VERBALIZERS.copy())
        task_perm = build_perm_task(seed=args.seed + r*1000)
        print(f"\n--- Repetition {r+1}/{args.repeat} ---")
        delta_kl = get_tasks_kl(
            scorer, data, task_true, task_perm, max_examples=args.max_kl_examples
        )
        kl_list.append(delta_kl)
        print(f"Δ_KL (sym) rep={r+1}: {delta_kl:.4f}")

        successes = []
        gaps = []
        for k in ks:
            succ, gap = loglikelihood_ratio_success(
                scorer,
                pool=data,
                task_true=task_true,
                task_perm=task_perm,
                k=k,
                trials=args.trials,
                eps=args.eps,
                flip_labels=args.flip,
                balanced=args.balanced,
                seed=args.seed + r*1000 + k*13,
            )
            print(f"rep={r+1} k={k:>2} | succ≈ {succ*100:5.1f}% | gap={gap:7.2f}")
            successes.append(succ)
            gaps.append(gap)
        succ_mat.append(successes)
        gap_mat.append(gaps)

    succ_mat = np.array(succ_mat)
    gap_mat = np.array(gap_mat)
    kl_arr = np.array(kl_list)

    succ_mean, succ_std = succ_mat.mean(axis=0), succ_mat.std(axis=0)
    gap_mean, gap_std   = gap_mat.mean(axis=0), gap_mat.std(axis=0)
    kl_mean, kl_std     = kl_arr.mean(), kl_arr.std()

    print(f"\n=== KL Summary ===")
    print(f"Δ_KL mean ≈ {kl_mean:.4f} ± {kl_std:.4f} (std over {args.repeat} runs)")

    plt.figure(figsize=(12,8))

    ax1 = plt.gca()
    ax1.errorbar(ks, succ_mean, yerr=succ_std, marker="o", color="blue", capsize=4, label="Success probability")
    ax1.set_xlabel("k (number of shots)")
    ax1.set_ylabel("Success probability", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.errorbar(ks, gap_mean, yerr=gap_std, marker="s", color="green", capsize=4, label="E[LL_true - LL_perm]")
    ax2.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold (−log ε) = {threshold:.2f} (ε={args.eps})")
    ax2.set_ylabel("E[LL_true - LL_perm]", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(f"Lemma 1: Success prob & Gap vs k (trials: {args.repeat}{' flip' if args.flip else ''})")
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig(f"lemma1_{'flip' if args.flip else ''}{args.eps}.png")
    plt.show()
    
if __name__ == "__main__":
    main()
