from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random
from datasets import load_dataset


@dataclass
class Task:
    name: str
    verbalizers: Dict[int, str]  # {1: "pos label", 0: "neg label"}
    data: Optional[List[Tuple[str, int, str]]] = None  # (text, label_id, label_verbalizer)
    instr: str = None
    instr_blind: str = "Decide the category of each question. Answer with one of the provided examples if exist."  # instruction without label options




class TrecICLDataset:


    def __init__(self, split="train", seed=42):

        self.rng = random.Random(seed)
        ds = load_dataset("trec")[split]
        self.labels = ["DESC", "ENTY", "ABBR", "HUM", "LOC", "NUM"]
        self.verbalizers = {
            "DESC": "description",   # DESC
            "ENTY": "entity",        # ENTY
            "ABBR": "abbreviation",  # ABBR
            "HUM": "human",         # HUM
            "LOC": "location",      # LOC
            "NUM": "number",        # NUM
            "OTHER": "other"        
        }
        assert "coarse_label" in ds.column_names, "No such label key"
        label_key = "coarse_label"
        text_key = "text"
        self.data = []
        for row in ds:
            y = int(row[label_key])
            self.data.append((row[text_key], y, self.verbalizers[self.labels[y]]))




    def get_original_dataset(self):
        return self.data.copy()

    def get_original_task(self):
        verbalizers = {i: self.verbalizers[label] for i, label in enumerate(self.labels)}
        task = Task(name="trec", verbalizers=verbalizers, data=self.data.copy())
        task.instr = make_task_instr(task)
        return task
    
    def build_binary_subtask(self, pos, neg="OTHER"):
        tname = f"{pos.upper()}_vs_{neg.upper()}"
        pos = self.verbalizers[pos]
        neg = self.verbalizers[neg]
        task = Task(name=tname, verbalizers={1: pos, 0: neg})
        pairs = []
        data = self.data.copy()
        for text, _, name in data:
            if name == pos: y = 1
            elif neg == "other": y = 0
            elif name == neg: y = 0
            else: continue
            pairs.append((text, y, task.verbalizers[y]))
        task.data = pairs
        task.instr = make_task_instr(task)
        return task

    def sample_k_for_task(self, task, k_total=8, flip_prob=0):
        pairs = task.data.copy()

        cls0 = [p for p in pairs if p[1] == 0]
        cls1 = [p for p in pairs if p[1] == 1]
        k1 = k_total // 2
        k0 = k_total - k1
        if k_total == 1:
            return [self.rng.choice(pairs)]
        elif k_total == 0:
            return []
        elif len(cls0) < k0 or len(cls1) < k1:
            raise ValueError("Try select more samples.")
        out = self.rng.sample(cls0, k0) + self.rng.sample(cls1, k1)
        self.rng.shuffle(out)
        if flip_prob > 0:
            tmp = []
            for ex in out:
                if len(ex) == 3:
                    q, y, z = ex
                else:
                    q, y = ex
                    z = None
                if self.rng.random() < flip_prob:
                    y = 1 - y
                    z = task.verbalizers[y]
                tmp.append((q, y, z) if z is not None else (q, y))
            out = tmp
        return out

def make_task_instr(task):
    if task.name == "trec":
        instr = (
            "You are given a question. "
            "Choose its category from the following options:\n"
            "- abbreviation\n"
            "- description\n"
            "- entity\n"
            "- human\n"
            "- location\n"
            "- number\n\n"
        )
    else:
        instr = (
            "You are given a question. "
            "Choose its category from the following options:\n"
        )
        for v in task.verbalizers.values():
            instr += f"- {v.lower()}\n"
        instr += "\n"
    return instr

if __name__ == "__main__":
    trec = TrecICLDataset(split="train", seed=42)
    original = trec.get_original_dataset()
    task = trec.build_binary_subtask("NUM", "LOC")
    fewshot = trec.sample_k_for_task(task, k_total=8)
    print("Original size:", len(original))
    print("Subtasks:", task.name, task.verbalizers)
    print(f"Few-shot samples for {task.name}:", fewshot)