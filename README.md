<p align="center">
  <img src="assets/raamichess.png" alt="Chessmaxx" width="350">
</p>

I like chess and I like LLM's.

Unfortunately, LLM’s kinda suck at chess.

Things start to stop working pretty fast when playing against them; hallucinations are rampant, they constantly need to be reminded of legal moves, they forget their own pieces, and they make a lot of …… questionable moves VERY overconfidently.

Past the opening, it all falls down (like the song).

So, I wanted challenge myself and see how much better I can make one (but theres a catch).

I only have access to an RTX 3060 Ti. Not a bad GPU for Fortnite, but only 8 GB of VRAM means I will have to maximize and optimize every little detail.

Most clusters and cloud trainings get to spend their resources on the knits of RL and SFT, but I think infra is overlooked and will start to get more and more important.

While Chessmaxxing, I will take a small open-weight language model and train it for chess using Stockfish supervision (I was very inspired by Meta’s recent success with Glimmer using Logit Distillation) while remembering the VRAM limit hanging over my head.

## Evaluation metrics

Metrics I want to use to measure progress in improvement on the evaluation harness:

- Elo
- First-try legal move rate
- Blunder rate
- Game completion rate
- And most importantly, Stockfish primary move agreement rate

## Infrastructure metrics

On the infra side, I want to track:

- Peak VRAM usage
- GPU utilization
- Batch size
- Training throughput
- GPU capability improvement
- And more

My objective isn’t to try to mog Stockfish in chess.

My goal is to see if I can ascend the model while being on a budget.

Will be documenting my thought process and findings :)

## Evaluation harness

The first Chessmaxx milestone is a reproducible frozen-position evaluation. The harness gives a model a FEN position, asks for exactly one UCI move, checks the first generated token, and compares legal moves with fixed-budget Stockfish analysis.

It currently reports:

- Parse rate
- First-try legal move rate
- Stockfish top-1 and top-k agreement
- Average centipawn regret
- 100, 300, and 500 centipawn blunder rates
- Mean, p50, and p95 generation latency

Agreement rates use every position as the denominator, so malformed and illegal responses are penalized. Centipawn regret and blunder rates use scored legal moves as their denominator. Raw model output and every position-level result remain in the JSON report for auditing.

### Setup

Python 3.11 or newer and a UCI-compatible Stockfish executable are required. Install a CUDA-compatible PyTorch build for your machine first, then install Chessmaxx:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,model]'
```

Run the smoke evaluation:

```bash
chessmaxx-eval positions \
  --model Qwen/Qwen3-0.6B-Base \
  --dataset data/eval/smoke.jsonl \
  --stockfish /path/to/stockfish \
  --output artifacts/evals/qwen3-base-smoke.json \
  --cache artifacts/stockfish-cache.json \
  --batch-size 8 \
  --nodes 50000 \
  --multipv 3
```

Use `--limit N` for a quick partial run. Model generation is greedy, and Stockfish defaults to one thread to make comparisons as stable as possible.

## Model baseline

Phase 2 records the unmodified model's chess ability before any training. Qwen3-0.6B Base is the primary baseline, Qwen3-0.6B measures the effect of general post-training, and SmolLM2-360M is a smaller-vocabulary control. Their complete settings live in `configs/baseline/`.

First, freeze an evaluation set from PGN games:

```bash
chessmaxx-eval sample-pgn \
  --pgn data/raw/games.pgn \
  --output data/eval/baseline-v1.jsonl \
  --count 1000 \
  --seed 2026 \
  --minimum-ply 8 \
  --max-per-game 4
```

Sampling is deterministic, capped per source game, and balanced across opening, middlegame, and endgame positions when enough positions are available.

Then run the primary baseline:

```bash
chessmaxx-eval baseline \
  --profile configs/baseline/qwen3-0.6b-base.toml \
  --dataset data/eval/baseline-v1.jsonl \
  --stockfish /path/to/stockfish \
  --cache artifacts/stockfish-cache.json \
  --output artifacts/evals/qwen3-0.6b-base.json
```

The run creates both the final report and an append-only `.progress.jsonl` journal beside it. If the process is interrupted, rerunning the identical command restores completed positions. A journal is accepted only when its dataset, model revision, engine identity, profile, and runtime settings match the new run.

Each report records:

- Requested and resolved model revision
- Architecture, tokenizer, parameter count, vocabulary size, and dtype
- PyTorch, Transformers, Python, CUDA, Chessmaxx, and Git versions
- Dataset and profile SHA-256 fingerprints
- Stockfish identity and search settings
- Attempted batch sizes and out-of-memory retries
- Peak allocated and reserved VRAM
- Position and token throughput
- Every raw model output and position-level chess result

If a batch exceeds available VRAM, the evaluator clears cached CUDA memory and bisects it automatically. The attempted batch and retry count remain in telemetry, so this fallback is visible in the report.

## Tiny supervised fine-tuning

Phase 3 proves the complete learning path on only 100 positions before spending hours on a larger run. Stockfish supplies ranked legal moves, Qwen3-0.6B Base learns the primary move with LoRA, and a deliberately in-sample check answers one narrow question: can the system make the model memorize the supervision it was given?

This is a pipeline diagnostic, not a chess-strength result. Held-out Stockfish agreement and Elo come later. A high memorization score shows that data, labels, masking, optimization, checkpoint saving, adapter loading, and generation all agree on the task.

### Build the teacher dataset

Install the training dependencies on the CUDA machine after installing the appropriate PyTorch build for that machine:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,train]"
```

Create at least 100 training examples from a PGN. Splits are assigned by source game, so positions from one game cannot leak between training and validation:

```powershell
chessmaxx-data build `
  --pgn data/raw/games.pgn `
  --output data/processed/tiny-sft-v1.jsonl `
  --count 125 `
  --seed 2026 `
  --minimum-ply 8 `
  --max-per-game 4 `
  --validation-fraction 0.1 `
  --stockfish C:\path\to\stockfish.exe `
  --nodes 50000 `
  --multipv 3
```

The command writes the versioned JSONL dataset, a SHA-256 manifest, and an append-only label journal. Rerunning the same command resumes completed Stockfish work. Changing the positions, split assignments, engine identity, search settings, schema, or prompt invalidates the journal instead of mixing incompatible labels.

### Run and resume tiny SFT

The checked-in profile pins the Qwen revision validated in Phase 2 and uses BF16, response-only labels, greedy sequence packing, gradient checkpointing, a physical batch size of 2, eight examples per optimizer step through gradient accumulation, and rank-16 LoRA over all linear layers:

```powershell
chessmaxx-train `
  --profile configs/train/tiny-sft-qwen3-0.6b.toml `
  --dataset data/processed/tiny-sft-v1.jsonl `
  --output-dir artifacts/training/tiny-sft-qwen3-0.6b
```

To resume an interrupted run, point at one of the saved checkpoint directories:

```powershell
chessmaxx-train `
  --profile configs/train/tiny-sft-qwen3-0.6b.toml `
  --dataset data/processed/tiny-sft-v1.jsonl `
  --output-dir artifacts/training/tiny-sft-qwen3-0.6b `
  --resume-from-checkpoint artifacts/training/tiny-sft-qwen3-0.6b/checkpoints/checkpoint-25
```

The run refuses to silently train on CPU. It saves checkpoints under `checkpoints/`, the final adapter and tokenizer under `final/`, and `training-report.json` at the run root. The report fingerprints the profile and dataset and records model revision, parameter counts, package versions, GPU identity, peak VRAM, wall time, and input-token throughput.

Packing concatenates complete examples and never truncates a target. Prompt tokens and padding use ignored labels, so only the move and EOS token contribute directly to loss. This first implementation uses ordinary causal attention across packed-example boundaries; block-diagonal attention isolation is a later optimization to benchmark explicitly.

### Verify memorization

Reload the final adapter and evaluate the same 100 training examples:

```powershell
chessmaxx-memorize `
  --profile configs/train/tiny-sft-qwen3-0.6b.toml `
  --dataset data/processed/tiny-sft-v1.jsonl `
  --adapter-dir artifacts/training/tiny-sft-qwen3-0.6b/final `
  --output artifacts/evals/tiny-sft-memorization.json `
  --batch-size 8
```

The report keeps every raw response and separately measures parse rate, legal-move rate, target-move accuracy, and exact-response accuracy. If target accuracy remains low, the tiny experiment should be debugged or deliberately overfit further before scaling the dataset.

### Position format

Evaluation sets use one JSON object per line:

```json
{
  "position_id": "game-42-ply-31",
  "game_id": "game-42",
  "ply": 31,
  "phase": "middlegame",
  "fen": "...",
  "teacher_moves": [
    {"move": "e2e4", "score_cp": 42},
    {"move": "d2d4", "score_cp": 35}
  ],
  "metadata": {"source": "fixture"}
}
```

Teacher moves are optional because the harness can calculate them with Stockfish. When supplied, they must be legal, unique, and sorted from highest to lowest score. Positions must be valid and non-terminal, and position IDs must be unique within a dataset.

### Development

Run the test suite with:

```bash
pytest
```

Downloaded PGNs, processed datasets, model checkpoints, evaluation reports, and Stockfish caches are intentionally excluded from Git.
