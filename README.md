<p align="center">
  <img src="assets/raamichess.png" alt="Chessmaxx" width="350">
</p>

I like chess and I like LLM.

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
