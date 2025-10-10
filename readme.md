# In-Context Learning Theorem 1 Experiments

This repository implements empirical experiments for validating **Lemma 1** and **Theorem 1** from  
*“The Learnability of In-Context Learning”* (Wies et al., 2024).  

[Report](https://github.com/WitchPuff/icl/blob/main/report.pdf)

---

## 📂 Repository Structure

| File/Folder | Description |
|--------------|-------------|
| **`dataset.py`** | Defines the **TREC dataset** class and helper functions for building binary subtasks (e.g., *NUM vs OTHERS*, *LOC vs OTHERS*). Also includes utilities for balanced k-shot sampling. |
| **`main.py`** | Main experiment script. Runs evaluation with different prompt types, random label flips, and concentration thresholds. |
| **`qa_0_1e-2/`** | Results using QA-style prompts with **no random label flipping**. Concentration threshold `ε = 1e-2`, margin success threshold `δ = 1e-2`. |
| **`qa_0.5_1e-2/`** | Results using QA-style prompts with **0.5 probability random label flipping**. Same thresholds `ε = 1e-2`, `δ = 1e-2`. |
| report.pdf | The report of this project. |

---

## ⚙️ Installation

```bash
git clone https://github.com/witchpuff/icl.git
cd icl
pip install -r requirements.txt
python dataset.py
python main.py
```