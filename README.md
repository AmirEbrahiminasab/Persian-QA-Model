# Fine-Tuning Language Models for Persian Question Answering

This repository contains the code and results for fine-tuning and evaluating **XLM-RoBERTa** and **ParsBERT** models on the **PersianQA** dataset for question answering tasks.

---

## 🎯 Objective: Evaluating and Improving Persian QA Models

This project reports on the fine-tuning of two prominent language models — **XLM-RoBERTa** and **ParsBERT** — on the PersianQA dataset. The primary goals were:

- To establish baseline performance of pre-trained models on a Persian question-answering task.  
- To enhance model performance through a targeted fine-tuning process.  
- To compare the effectiveness of the two architectures both before and after fine-tuning.  

---

## 🔍 Methodology

### 🧠 Models

- **XLM-RoBERTa**: `pedramyazdipoor/persian_xlm_roberta_large`  
- **ParsBERT**: `pedramyazdipoor/parsbert_question_answering_PQuAD`

### 📚 Dataset

- **PersianQA**: `SajjadAyoubi/persian_qa`

### 📏 Evaluation Metrics

- **Exact Match (EM)**: Measures the percentage of predictions that match the ground truth answers exactly.  
- **F1-Score**: The harmonic mean of precision and recall, measuring token overlap.

---

## ⚙️ Fine-Tuning Parameters

| Argument              | Value     |
|-----------------------|-----------|
| Learning Rate         | 2 × 10⁻⁵  |
| Training Epochs       | 10        |
| Train Batch Size      | 8         |
| Evaluation Batch Size | 8         |
| Weight Decay          | 0.01      |
| Scheduler Type        | Cosine    |
| Warmup Ratio          | 0.1       |
| Best Model Metric     | F1-Score  |

---

## 📈 Performance Results

### XLM-RoBERTa Performance

**Comparison of XLM-RoBERTa Before and After Fine-Tuning**

| Model Status                  | Exact Match | F1-Score |
|------------------------------|-------------|----------|
| Base Model (Before)          | 45.81%      | 41.50%   |
| Fine-Tuned (10 Epochs)       | 65.27%      | 82.63%   |
| Fine-Tuned with LoRA (8 Ep.) | 69.90%      | 84.85%   |

**Highlights:**

- **Exact Match Gain**: +24 points (fine-tuned), +28.4 points (LoRA)  
- **F1-Score Gain**: +37 points (fine-tuned), +39 points (LoRA)  

#### LoRA Efficiency (XLM-RoBERTa)

- **Training Time**: ~3 hours (1 hour faster than full fine-tuning)  
- **Parameter Efficiency**: Only 0.281% updated  
- **Trainable Params**: 1,574,914  
- **Total Params**: 560,417,796  

---

### ParsBERT Performance

**Comparison of ParsBERT Before and After Fine-Tuning**

| Model Status                  | Exact Match | F1-Score |
|------------------------------|-------------|----------|
| Base Model (Before)          | 5.27%       | 9.69%    |
| Fine-Tuned (10 Epochs)       | 55.59%      | 71.97%   |
| Fine-Tuned with LoRA         | 50.75%      | 64.34%   |

**Highlights:**

- **Exact Match Jump**: +50 percentage points  
- **F1-Score Surge**: +62 points with full fine-tuning  

#### LoRA Efficiency (ParsBERT)

- **Training Time**: ~1 hour (22 minutes faster than full fine-tuning)  
- **Parameter Efficiency**: Only 0.3631% updated  
- **Trainable Params**: 591,362  
- **Total Params**: 162,843,652  

---

## 🧪 Final Results: Head-to-Head Comparison

| Fine-Tuned Model | Exact Match | F1-Score |
|------------------|-------------|----------|
| XLM-RoBERTa      | 69.90%      | 84.84%   |
| ParsBERT         | 55.59%      | 71.97%   |

---

## 📊 Experiment Analysis

### ParsBERT: Analysis on Data Subsets

| Analysis Case       | Exact Match | F1-Score |
|---------------------|-------------|----------|
| Has Answer          | 44.70%      | 68.22%   |
| No Answer           | 78.14%      | 78.14%   |
| Longer than Avg.    | 38.56%      | 69.80%   |
| Shorter than Avg.   | 53.01%      | 68.88%   |

### XLM-RoBERTa: Analysis on Data Subsets *(LoRA Model)*

| Analysis Case       | Exact Match | F1-Score |
|---------------------|-------------|----------|
| Has Answer          | 62.06%      | 83.42%   |
| No Answer           | 88.17%      | 88.17%   |
| Longer than Avg.    | 49.22%      | 81.88%   |
| Shorter than Avg.   | 62.95%      | 80.20%   |

---

## 🤝 Comparative Analysis of Model Capabilities

| Analysis Case       | XLM-RoBERTa (EM) | XLM-RoBERTa (F1) | ParsBERT (EM) | ParsBERT (F1) |
|---------------------|------------------|------------------|----------------|----------------|
| Has Answer          | 62.06%           | 83.42%           | 44.70%         | 68.22%         |
| No Answer           | 88.17%           | 88.17%           | 78.14%         | 78.14%         |
| Longer than Avg.    | 49.22%           | 81.88%           | 38.56%         | 69.80%         |
| Shorter than Avg.   | 62.95%           | 80.20%           | 53.01%         | 68.88%         |

---

## 📝 Key Takeaways

- **Overall Superiority**: XLM-RoBERTa consistently outperforms ParsBERT across all evaluation scenarios.  
- **Unanswerable Questions**: Both models perform well, but XLM-RoBERTa (LoRA) leads by ~10 points.  
- **Answer Length Impact**: While both models do better on shorter answers, XLM-RoBERTa retains strong performance even on long responses.

---

## 🚀 How to Run This Repo

### 1. Setup

```bash
git clone https://github.com/your-username/persian-qa.git
cd persian-qa
pip install -r requirements.txt
````

### 2. Training

**Full Fine-Tuning:**

```bash
python scripts/train.py \
    --model_name "pedramyazdipoor/persian_xlm_roberta_large" \
    --output_dir "./models/xlm-roberta-finetuned"
```

**LoRA Fine-Tuning:**

```bash
python scripts/train.py \
    --model_name "pedramyazdipoor/persian_xlm_roberta_large" \
    --output_dir "./models/xlm-roberta-lora" \
    --use_lora
```

*Replace `model_name` and `output_dir` accordingly for ParsBERT.*

### 3. Running Experiments

```bash
python scripts/run_experiments.py --model_path "./models/xlm-roberta-lora"
```

This will output performance metrics across subsets (has/no answer, long/short answers).

---

## 📄 License

This project is licensed under the **MIT License**.
