# Nanochat NVFP4 Quickstart

Quick setup guide for NVFP4 training with nanochat. Requires Blackwell GPU (GB200, B100, B200).

## Prerequisites

Install dependencies:
```bash
pip install --user wandb rustbpe tiktoken datasets tabulate
```

Install patched TorchAO (with zero-scale fix):
```bash
pip install -e /path/to/ao --no-build-isolation
```

## Setup (One-time)

```bash
cd /path/to/nanochat

# Download dataset (minimum 1 shard, ~100MB)
python -m nanochat.dataset -n 1

# Train tokenizer (~11 seconds)
python -m scripts.tok_train
```

## Training

First compilation takes a few minutes.

### Quick test (~2 min)
```bash
python -m scripts.base_train --nvfp4 --depth=4 --num-iterations=10 --device-batch-size=4 --max-seq-len=512 --window-pattern=L --eval-every=-1 --core-metric-every=-1 --sample-every=-1
```

### Meaningful model (~5 min)
```bash
python -m scripts.base_train --nvfp4 --depth=12 --num-iterations=100 --device-batch-size=8 --window-pattern=L
```

### GPT-2 scale (~2 hours on 8xGPU)
```bash
# Download more data first
python -m nanochat.dataset -n 170

torchrun --standalone --nproc_per_node=8 -m scripts.base_train --nvfp4 --depth=24 --device-batch-size=16
```

## Key flags

| Flag | Description | Default |
|------|-------------|---------|
| `--depth` | Model depth (layers). GPT-2 ~ depth 24 | 20 |
| `--num-iterations` | Training steps | auto |
| `--device-batch-size` | Per-GPU batch size. Reduce if OOM | 32 |
| `--max-seq-len` | Context length | 2048 |
| `--window-pattern` | Attention pattern. Use `L` without FA3 | SSSL |
| `--eval-every` | Eval frequency (-1 to disable) | 250 |

## Troubleshooting

**OOM errors** - Reduce `--device-batch-size` (try 16, 8, 4, 2, 1)

**"Flash Attention 3 not available"** - Use `--window-pattern=L`

**"FileNotFoundError: tokenizer.pkl"** - Run `python -m scripts.tok_train`
