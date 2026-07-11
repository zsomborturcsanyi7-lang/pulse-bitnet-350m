# MicroLanguageSwarm — Pulse 350M BitNet Language Model

**Version:** 1.0  
**Author:** Zsombi & Hermes Agent (Nous Research)  
**Status:** Active development

---

## Description

The MicroLanguageSwarm project implements a **350 million parameter, 1.58-bit BitNet language model** (Pulse). The model is highly efficient: only **~84 MB** in BitNet format and runs on CPU at **30+ tokens/second**. The project includes the full Kaggle 4-session training pipeline, RLAIF fine-tuning, and training optimized for Hungarian language data.

---

## File Structure

```
MicroLanguageSwarm/
│
├── pulse_350m_bitnet.py      # Pulse 350M model architecture (BitLinear + GQA + RoPE)
├── train_bitnet_local.py     # Local GPU training pipeline
├── kaggle_session1.py        # Kaggle Session 1 — data preparation
│
├── distill_bitnet.py         # Knowledge distillation from teacher model
├── distill_cross.py          # Cross-distillation
├── gen_teacher_data.py       # Teacher model data generation
│
├── rlaif_train.py            # RLAIF fine-tuning
├── rlaif_batched.py          # Batched RLAIF training
├── rlaif_kw.py               # RLAIF keyword optimization
│
├── persistent_train.py       # Persistent GPU training (with watchdog)
├── continue_train.py         # Continue training from checkpoint
├── fast_train.py             # Accelerated training
├── train_all_shards.py       # Train all 4 shards
├── watchdog.py               # GPU watchdog (TDR protection)
│
├── query_bitnet.py           # Model query / inference
├── test_generate.py          # Text generation test
├── test_bitnet.py            # BitNet unit test
├── test_rlaif.py             # RLAIF test
├── check_model.py            # Model verification
├── check_ckpt.py             # Checkpoint verification
├── check_datasets.py         # Dataset verification
│
├── benchmark.py              # Benchmark measurement
├── bench_speed.py            # Speed benchmark
├── quick_test.py             # Quick validation test
├── test_cuda_remote.py       # CUDA remote test
│
├── download_hu_data.py       # Hungarian data download
│
├── run_training.vbs          # VBS launcher script
├── run_pt.bat                # Persistent training launcher
├── run_s3.bat                # Session 3 launcher
│
├── teacher_model/            # Teacher model (fine-tuned NEURA)
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
│
├── bitnet_kaggle_data/       # Kaggle data
│   ├── shard_0.pt ... shard_3.pt
│   └── dataset-metadata.json
│
├── bitnet_kernel_push/       # Kaggle kernel push files
├── session1_kernel/          # Session 1 kernel output
│
└── __pycache__/              # Python cache
```

---

## Usage

### Loading and Running the Model

```bash
# Environment setup
pip install torch sentencepiece

# Load model
python query_bitnet.py

# Quick test
python quick_test.py
```

### Local Training

```bash
# Local GPU training
python train_bitnet_local.py

# Persistent training (with watchdog protection)
run_pt.bat

# Start Session 3
run_s3.bat
```

### Kaggle Training Pipeline

1. **Session 1:** Data preparation — `kaggle_session1.py`
2. **Session 2:** Start training — `train_all_shards.py`
3. **Session 3:** Continuation — `run_s3.bat`
4. **Session 4:** RLAIF fine-tuning — `rlaif_train.py`

### Knowledge Distillation

```bash
# Generate teacher data
python gen_teacher_data.py

# Distillation
python distill_bitnet.py

# Cross-distillation
python distill_cross.py
```

### RLAIF Fine-Tuning

```bash
python rlaif_train.py
# or batched version:
python rlaif_batched.py
```

---

## Architecture

| Component | Specification |
|-----------|-------------|
| **Type** | 1.58-bit BitNet (BitLinear layers) |
| **Parameters** | 350M / ~84 MB |
| **Attention** | GQA (Grouped Query Attention) |
| **Normalization** | SubLN (Sub-Layer Normalization) |
| **Activation** | ReLU² |
| **Position** | RoPE (Rotary Position Embedding) |
| **Bias** | None (bias=False) |
| **Speed** | CPU: 30+ tok/sec |

---

## Dependencies

- **Python** 3.10+
- **PyTorch** 2.0+ (CUDA optional)
- **SentencePiece** (tokenizer)
- **NumPy**

---

## Developer

Zsombi & Hermes Agent (Nous Research)
