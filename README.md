# **Handwritten Math Expression Recognition (HMER)**

A complete end-to-end system for handwritten mathematical expression recognition based on CNN + Transformer, supporting training, evaluation and Web demo.

## 📌 Project Overview

This project implements an end-to-end handwritten mathematical expression recognition (HMER) system that converts handwritten math expression images into LaTeX code.

**The system is designed as a vision-to-sequence model and covers the full engineering pipeline:**

Dataset construction and preprocessing

CNN + Transformer encoder–decoder modeling

Training, fine-tuning, and evaluation (token-level & formula-level)

Advanced optimization strategies (oversampling, long-formula finetuning, weighted loss, beam search, etc.)

Web-based interactive demo for real-world usage

**The project is suitable for:**

Course projects / capstone projects

Research prototyping in OCR / HMER

Engineering-oriented deep learning practice


## ✨ Features

CNN + Transformer architecture for structured math recognition

Token-level & formula-level evaluation

Special optimization for long mathematical expressions

Weighted loss for digits / structure / variables / units

Beam Search decoding

Web demo with image upload & LaTeX rendering

Clean, modular, and extensible codebase


## 🧠 Model Architecture

`Input Image
    ↓
CNN Encoder (feature extraction)
    ↓
Transformer Encoder (visual sequence modeling)
    ↓
Transformer Decoder (autoregressive LaTeX generation)
    ↓
LaTeX Token Sequence`


**Key characteristics:**

·Encoder extracts 2D visual features and converts them into a sequence

·Decoder generates LaTeX tokens autoregressively

·Attention mechanism models long-range dependencies

·Designed to handle nested structures like \frac, ^, _, \begin{matrix}


## 📁 Project Structure

`Handwritten-Math-Expression-Recognition-project/
│
├── config.py                 # Global configuration
├── requirements.txt
│
├── data/                      # Dataset (excluded from GitHub)
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   ├── train_long.txt
│   └── train_oversampled.txt
│
├── model/
│   ├── encoder.py
│   ├── decoder.py
│   └── model.py
│
├── utils/
│   ├── dataset.py
│   ├── vocab.py
│   └── ...
│
├── train/
│   ├── train.py
│   ├── train_long_finetune.py
│   ├── train_mixed_finetune.py
│   └── loss_utils.py
│
├── eval_test.py               # Evaluation script
├── predict.py                 # Single-image inference
│
├── web_demo/
│   └── app.py                 # Web application
│
├── checkpoints/               # Model checkpoints (ignored)
├── results/                   # Evaluation outputs
└── README.md`


## ⚙️ Environment Setup

### 1. Clone Repository

`git clone https://github.com/YourUsername/Handwritten-Math-Expression-Recognition-project.git
cd Handwritten-Math-Expression-Recognition-project`

### 2. Create Python Environment

`conda create -n hmer python=3.9
conda activate hmer`

### 3. Install Dependencies

`pip install -r requirements.txt`


## 📦 Dataset Preparation

### 1. Dataset Description

The project uses ICDAR-style handwritten math datasets.

Each annotation file follows the format:

`relative/image/path.jpg<TAB>latex tokens (space-separated)`


Example:

`data/icdar_raw/train_images/train_1234.jpg    \frac { 1 } { x } + 2`

### 2. Dataset Download

Due to size limitations, datasets are not included in the repository.

Please download the dataset from the following cloud link:

📥 Dataset Download (Baidu)
🔗 

After downloading, organize the data as:

`data/
├── icdar_raw/
│   └── train_images/
├── train.txt
├── val.txt
├── test.txt
├── train_long.txt
└── train_oversampled.txt`


⚠️ Paths in .txt files must match the actual image paths.


## 🚀 Training

### 1. Basic Training

`python train/train.py`


This performs standard teacher-forcing training using cross-entropy loss.

### 2. Long-Formula Fine-tuning

`python train/train_long_finetune.py`


Used to improve performance on long expressions (L > 20 tokens).

### 3. Mixed Fine-tuning (Recommended)

`python train/train_mixed_finetune.py`


**Key strategies:**

Mix normal & long-formula samples

Freeze encoder, fine-tune decoder

Weighted token loss

Label smoothing


## 📊 Evaluation

**Run Evaluation on Test Set**
`python eval_test.py`


Metrics reported:

Average loss

Token accuracy

Formula accuracy

Formula accuracy by length bucket

Near-miss (almost-correct) statistics

Evaluation results are saved to:

`results/
├── test_summary.txt
└── test_samples.txt`


## 🔍 Single Image Inference

`python predict.py --image path/to/image.jpg`


Outputs:

Predicted LaTeX code

Rendered math expression


## 🌐 Web Demo

Start Web Application
`cd web_demo
python app.py`


Then open in browser:

http://127.0.0.1:7860


Features:

·Upload handwritten math image

·Display original image

·Render predicted LaTeX

·Download LaTeX code

·History of recognition results


## 📈 Current Performance (Stage Summary)

Metric	            Value
Token Accuracy	    ~93–94%
Formula Accuracy	~42%
Short formulas	    >58%
Long formulas	    ~32%

Long expressions remain the primary challenge.


## 🔧 Implemented Optimization Strategies

Data oversampling for long formulas

Long-formula fine-tuning

Encoder freezing

Weighted token loss

Beam search decoding

Near-miss error analysis

Photo-style image preprocessing


## 🔮 Future Work

Scheduled Sampling

Structural constraints during decoding

Grammar-aware loss

AST-based LaTeX modeling

Further robustness for real-world photos


## 📜 License

This project is for academic and research use.
Please cite or reference appropriately if used in publications.


## 🙋 Author

Author: Allen
Project Type: Handwritten Mathematical Expression Recognition
Status: Actively developing
