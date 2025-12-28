# SEW (KDD 2026)

This repository contains the official implementation of the paper **"SEW: Strengthening Robustness of Black-box DNN Watermarking via Specificity Enhancement" (KDD 2026)**.

## 📖 Introduction

Black-box DNN watermarking protects intellectual property by embedding trigger sets. However, existing methods often suffer from **low specificity**—meaning the watermark is activated not just by the exact key, but by many "approximate keys" (noisy versions). This vulnerability allows attackers to reverse-engineer the key and remove the watermark.

**SEW (Specificity-Enhanced Watermarking)** introduces a novel training paradigm that:

1. **Quantifies Specificity:** We propose a metric to measure the "fuzziness" of the watermark key.
2. **Enhances Specificity:** We use an adaptive noise optimization strategy to generate "cover samples" (approximate keys) during training and suppress their activation.

This results in a watermark that is highly specific to the original key, significantly improving robustness against state-of-the-art removal attacks (e.g., Dehydra, MOTH, FeatureRE).

## 🛠️ Installation

### 1. Create and Activate Conda Environment

```bash
conda create -n sew python=3.10
conda activate sew
```

### 2. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

The workflow consists of two main steps: **Training** the watermarked model (embedding) and **Evaluating** the specificity of the watermark.

For ease of reproduction, we provide ready-to-use Bash scripts in the `scripts/` directory.

**⚠️ Important Note**

Before running any scripts, please open the `.sh` files and update the following parameters to match your local environment:

* `root`: The actual path to your dataset.
* `save_path`: The directory where you want to save model checkpoints and logs.

---

### Step 1: Embedding Watermark (Training)

Use the `scripts/run_sew.sh` script to train the watermarked model. By default, this script is configured to train **SEW-Post** on **CIFAR-10** using a **VGG16** architecture.

```bash
# Run the training script after configuring paths
bash scripts/run_sew.sh
```

Upon completion, the best and final model checkpoints will be saved to your specified `save_path` (e.g., `final.pth` and `best.pth`).

### Step 2: Measuring Specificity

After training, use the `scripts/run_spec.sh` script to calculate the **Specificity (Spec)** metric. A *lower* Spec score indicates higher specificity (better security).

```bash
# Ensure 'load_path' in the script points to the model generated in Step 1
bash scripts/run_spec.sh
```

This script loads the trained model and dynamically calculates the epsilon (`eps`) noise boundary. The final output `eps` is the Specificity score.

## 📊 Results Summary

Our experiments demonstrate a strong correlation between **watermark specificity** and **robustness**. By optimizing the watermark to be highly specific (low `Spec` score), SEW effectively resists state-of-the-art removal attacks.

As demonstrated in the paper, SEW-Post significantly reduces the specificity score compared to baselines:

| Method | Dataset | CDA | WACC | Specificity |
| --- | --- | --- | --- | --- |
| **SEW-Pre** (Baseline) | CIFAR-10 | 93.03% | 100% | 0.3569 |
| **SEW-Post** (Ours) | CIFAR-10 | 92.79% | 100% | **0.0364** |

High specificity translates to strong defense, making the watermark much harder to reverse-engineer. While the baseline fails against state-of-the-art removal attacks, **SEW-Post** maintains near-perfect watermark verification accuracy:

| Method | Dehydra | MOTH | FeatureRE |
| --- | --- | --- | --- |
| **SEW-Pre** (Baseline) | 2.67% | 7.67% | 8.00% |
| **SEW-Post** (Ours) | **98.67%** | **97.67%** | **98.33%** |

> **Note:** For more comprehensive experimental results, including comparisons with 10+ baselines and detailed specificity analysis on NLP tasks, please refer to our paper.


## 🔗 Citation

If you find this code or paper useful for your research, please cite our work.