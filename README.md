# MicroLanguageSwarm — Pulse 350M BitNet Nyelvi Modell

**Verzió:** 1.0  
**Szerző:** Zsombi (AI asszisztens segítségével)  
**Státusz:** Aktív fejlesztés

---

## Leírás

A MicroLanguageSwarm projekt egy **350 millió paraméteres, 1.58-bites BitNet nyelvi modellt** (Pulse) valósít meg. A modell rendkívül hatékony: mindössze **~84 MB** méretű BitNet formátumban és CPU-n **30+ token/másodperc** sebességgel fut. A projekt tartalmazza a teljes Kaggle 4-szakaszos training pipeline-t, RLAIF finomhangolást, és magyar nyelvi adatokra optimalizált tanítást.

---

## Fájlszerkezet

```
MicroLanguageSwarm/
│
├── pulse_350m_bitnet.py      # Pulse 350M modell architektúra (BitLinear + GQA + RoPE)
├── train_bitnet_local.py     # Helyi GPU training pipeline
├── kaggle_session1.py        # Kaggle Session 1 — adat előkészítés
│
├── distill_bitnet.py         # Tudás desztilláció tanár modellből
├── distill_cross.py          # Kereszt-desztilláció
├── gen_teacher_data.py       # Tanár modell adatgenerálás
│
├── rlaif_train.py            # RLAIF finomhangolás
├── rlaif_batched.py          # Batch-elt RLAIF training
├── rlaif_kw.py               # RLAIF kulcsszó optimalizálás
│
├── persistent_train.py       # Perzisztens GPU training (watchdoggal)
├── continue_train.py         # Training folytatása checkpointból
├── fast_train.py             # Gyorsított training
├── train_all_shards.py       # Mind a 4 shard betanítása
├── watchdog.py               # GPU watchdog (TDR védelem)
│
├── query_bitnet.py           # Modell lekérdezés / inference
├── test_generate.py          # Szöveggenerálás teszt
├── test_bitnet.py            # BitNet unit teszt
├── test_rlaif.py             # RLAIF teszt
├── check_model.py            # Modell ellenőrzés
├── check_ckpt.py             # Checkpoint ellenőrzés
├── check_datasets.py         # Adathalmaz ellenőrzés
│
├── benchmark.py              # Benchmark mérés
├── bench_speed.py            # Sebesség benchmark
├── quick_test.py             # Gyors ellenőrző teszt
├── test_cuda_remote.py       # CUDA távoli teszt
│
├── download_hu_data.py       # Magyar adatok letöltése
│
├── run_training.vbs          # VBS indító script
├── run_pt.bat                # Persistent training indító
├── run_s3.bat                # Session 3 indító
│
├── teacher_model/            # Tanár modell (finomhangolt NEURA)
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
│
├── bitnet_kaggle_data/       # Kaggle adatok
│   ├── shard_0.pt ... shard_3.pt
│   └── dataset-metadata.json
│
├── bitnet_kernel_push/       # Kaggle kernel push fájlok
├── session1_kernel/          # Session 1 kernel kimenet
│
└── __pycache__/              # Python cache
```

---

## Használat

### Modell betöltése és futtatása

```bash
# Környezet
pip install torch sentencepiece

# Modell betöltése
python query_bitnet.py

# Gyors teszt
python quick_test.py
```

### Training indítása lokálisan

```bash
# Helyi GPU training
python train_bitnet_local.py

# Perzisztens training (watchdog védelemmel)
run_pt.bat

# Session 3 indítása
run_s3.bat
```

### Kaggle training pipeline

1. **Session 1:** Adat előkészítés — `kaggle_session1.py`
2. **Session 2:** Training indítása — `train_all_shards.py`
3. **Session 3:** Folytatás — `run_s3.bat`
4. **Session 4:** RLAIF finomhangolás — `rlaif_train.py`

### Tudás desztilláció

```bash
# Tanár adatok generálása
python gen_teacher_data.py

# Desztilláció
python distill_bitnet.py

# Kereszt-desztilláció
python distill_cross.py
```

### RLAIF finomhangolás

```bash
python rlaif_train.py
# vagy batch-elt verzió:
python rlaif_batched.py
```

---

## Architektúra

| Komponens | Specifikáció |
|-----------|-------------|
| **Típus** | 1.58-bit BitNet (BitLinear rétegek) |
| **Paraméterek** | 350M / ~84 MB |
| **Attention** | GQA (Grouped Query Attention) |
| **Normalizáció** | SubLN (Sub-Layer Normalization) |
| **Aktiváció** | ReLU² |
| **Pozíció** | RoPE (Rotary Position Embedding) |
| **Bias** | Nincs (bias=False) |
| **Sebesség** | CPU: 30+ tok/sec |

---

## Függőségek

- **Python** 3.10+
- **PyTorch** 2.0+ (CUDA opcionális)
- **SentencePiece** (tokenizer)
- **NumPy**

---

## Fejlesztő

Zsombi (AI asszisztens segítségével) (AI asszisztens segítségével)
