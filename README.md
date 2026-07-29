# pulse-bitnet-350m

1.58-bit quantized Hungarian language model training pipeline.

## Overview & Purpose
pulse-bitnet-350m provides training scripts and data preprocessing routines for training a 1.58-bit (ternary) 350M parameter BitNet language model on Hungarian textual corpora.

## Key Features
- 1.58-bit (ternary) BitNet quantization layer implementations.
- Pretraining and Supervised Fine-Tuning (SFT) scripts.
- Text tokenization and dataset formatting utilities.

## Tech Stack & Dependencies
- **Framework**: PyTorch
- **Libraries**: HuggingFace Transformers, Datasets, Tokenizers
- **Language**: Python 3.10+

## Project Structure
```text
pulse-bitnet-350m/
├── train_350m_assistant.py
├── utils/
├── requirements.txt
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU environment

### Steps
```bash
git clone https://github.com/zsomborturcsanyi7-lang/pulse-bitnet-350m.git
cd pulse-bitnet-350m
pip install -r requirements.txt
```

## Usage Examples
```bash
python train_350m_assistant.py --epochs 3 --batch-size 16
```

## Status & License
Status: Experimental Training Pipeline.
License: MIT
