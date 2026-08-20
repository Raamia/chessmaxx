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

The correctness-reference profile pins the Qwen revision validated in Phase 2 and uses BF16, response-only labels, gradient checkpointing, a physical batch size of 2, eight examples per optimizer step through gradient accumulation, and rank-16 LoRA over all linear layers. It deliberately disables packing:

```powershell
chessmaxx-train `
  --profile configs/train/tiny-sft-qwen3-0.6b-unpacked.toml `
  --dataset data/processed/tiny-sft-v1.jsonl `
  --output-dir artifacts/training/tiny-sft-qwen3-0.6b-unpacked
```

To resume an interrupted run, point at one of the saved checkpoint directories:

```powershell
chessmaxx-train `
  --profile configs/train/tiny-sft-qwen3-0.6b-unpacked.toml `
  --dataset data/processed/tiny-sft-v1.jsonl `
  --output-dir artifacts/training/tiny-sft-qwen3-0.6b-unpacked `
  --resume-from-checkpoint artifacts/training/tiny-sft-qwen3-0.6b-unpacked/checkpoints/checkpoint-25
```

The run refuses to silently train on CPU. It saves checkpoints under `checkpoints/`, the final adapter and tokenizer under `final/`, and `training-report.json` at the run root. The report fingerprints the profile and dataset and records model revision, parameter counts, package versions, GPU identity, peak VRAM, wall time, and input-token throughput.

The original `tiny-sft-qwen3-0.6b.toml` profile is retained only as the naive-packing baseline. It concatenates complete examples and masks prompt loss, but its ordinary causal attention crosses example boundaries. Do not use it as a correctness reference.

### Verify memorization

Reload the final adapter and evaluate the same 100 training examples:

```powershell
chessmaxx-memorize `
  --profile configs/train/tiny-sft-qwen3-0.6b-unpacked.toml `
  --dataset data/processed/tiny-sft-v1.jsonl `
  --adapter-dir artifacts/training/tiny-sft-qwen3-0.6b-unpacked/final `
  --output artifacts/evals/tiny-sft-memorization-unpacked.json `
  --batch-size 8
```

The report keeps every raw response and separately measures parse rate, legal-move rate, target-move accuracy, and exact-response accuracy. If target accuracy remains low, the tiny experiment should be debugged or deliberately overfit further before scaling the dataset.

## Isolated sequence packing

The first packing experiment made training 2.66 times faster, but isolated evaluation exposed a perfect boundary failure: all 37 examples at the start of a pack were memorized, while all 63 later examples failed. Disabling packing restored 100% isolated accuracy.

| Layout | Records | Steps | Input tokens/s | Wall time | Legal moves | Target accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Naive causal packing | 37 | 100 | 1,308.34 | 135.40 s | 46% | 37% |
| Unpacked control | 100 | 260 | 491.36 | 348.32 s | 100% | 100% |

Phase 4 adds a third controlled profile. It retains the same greedy packs but supplies Qwen's SDPA backend with a boolean 4D block-causal mask. Tokens can attend only to earlier tokens from their own example, and rotary position IDs restart at every boundary.

```powershell
chessmaxx-train `
  --profile configs/train/tiny-sft-qwen3-0.6b-isolated.toml `
  --dataset data/processed/tiny-sft-v1.jsonl `
  --output-dir artifacts/training/tiny-sft-qwen3-0.6b-isolated

chessmaxx-memorize `
  --profile configs/train/tiny-sft-qwen3-0.6b-isolated.toml `
  --dataset data/processed/tiny-sft-v1.jsonl `
  --adapter-dir artifacts/training/tiny-sft-qwen3-0.6b-isolated/final `
  --output artifacts/evals/tiny-sft-memorization-isolated.json `
  --batch-size 8
```

The isolated run must match the unpacked control's 100-example memorization result before its throughput is treated as valid. A fast run with incorrect attention semantics is not an optimization.

### Held-out adapter evaluation

The memorization check is intentionally in-sample. Evaluate any saved adapter on the game-isolated validation split through the normal Stockfish harness:

```powershell
chessmaxx-eval adapter `
  --training-profile configs/train/tiny-sft-qwen3-0.6b-isolated.toml `
  --adapter-dir artifacts/training/tiny-sft-qwen3-0.6b-isolated/final `
  --dataset data/processed/tiny-sft-v1.jsonl `
  --split validation `
  --stockfish C:\path\to\stockfish.exe `
  --cache artifacts/stockfish-cache.json `
  --output artifacts/evals/tiny-sft-validation-isolated.json
```

The adapter report fingerprints the adapter tree, dataset, and training profile. The current ten-position validation split is a wiring check, not a statistically meaningful estimate of chess strength.

## Scaled supervised fine-tuning

Phase 5 moves from intentional memorization to held-out learning. The first scaled profile uses 900 training positions and 100 validation positions for five epochs. It keeps the model, LoRA shape, seed, batch settings, and isolated-packing implementation fixed.

Build a 1,200-position dataset with game-isolated validation and test splits:

```powershell
chessmaxx-data build `
  --pgn data/raw/games.pgn `
  --output data/processed/scaled-sft-v1.jsonl `
  --count 1200 `
  --seed 2026 `
  --minimum-ply 8 `
  --max-per-game 4 `
  --validation-fraction 0.1 `
  --test-fraction 0.1 `
  --minimum-train 900 `
  --minimum-validation 100 `
  --minimum-test 100 `
  --stockfish C:\path\to\stockfish.exe `
  --nodes 50000 `
  --multipv 3
```

The size checks run before Stockfish starts. With the Phase 3 PGN snapshot this deterministically yields 975 training, 114 validation, and 111 test positions.

Evaluate the pinned base model on validation before training:

```powershell
chessmaxx-eval labelled-base `
  --training-profile configs/train/scaled-sft-qwen3-0.6b-isolated.toml `
  --dataset data/processed/scaled-sft-v1.jsonl `
  --split validation `
  --stockfish C:\path\to\stockfish.exe `
  --cache artifacts/stockfish-cache.json `
  --output artifacts/evals/scaled-sft-validation-base.json
```

Run scaled training:

```powershell
chessmaxx-train `
  --profile configs/train/scaled-sft-qwen3-0.6b-isolated.toml `
  --dataset data/processed/scaled-sft-v1.jsonl `
  --output-dir artifacts/training/scaled-sft-qwen3-0.6b-isolated
```

The report records training and validation loss every 25 optimizer steps. All scheduled adapters are retained so chess capability can be measured throughout the run.

Evaluate every checkpoint on validation:

```powershell
chessmaxx-eval checkpoints `
  --training-profile configs/train/scaled-sft-qwen3-0.6b-isolated.toml `
  --run-dir artifacts/training/scaled-sft-qwen3-0.6b-isolated `
  --dataset data/processed/scaled-sft-v1.jsonl `
  --split validation `
  --stockfish C:\path\to\stockfish.exe `
  --cache artifacts/stockfish-cache.json `
  --output-dir artifacts/evals/scaled-sft-validation-checkpoints
```

Join systems and chess measurements into one curve artifact:

```powershell
chessmaxx-eval curve `
  --base-report artifacts/evals/scaled-sft-validation-base.json `
  --training-report artifacts/training/scaled-sft-qwen3-0.6b-isolated/training-report.json `
  --checkpoint-reports artifacts/evals/scaled-sft-validation-checkpoints `
  --output artifacts/evals/scaled-sft-validation-curve.json
```

Choose a checkpoint using validation only. Evaluate that checkpoint and the pinned base model once on the untouched test split for the final comparison. Do not use test results to tune the profile or select a checkpoint.

## Multi-PV policy distillation

Phase 6 uses more of the Stockfish supervision that is already present in each training record. Hard SFT learns only the primary move. The policy objective keeps up to three ranked moves and converts their centipawn scores into a soft target:

```text
p(move i) = exp((score_i - max_score) / temperature_cp) / Z
```

This is multi-PV policy distillation, not literal vocabulary-logit distillation. Stockfish supplies search utilities over legal moves rather than logits over Qwen's 151,936-token vocabulary. Naming that distinction explicitly keeps the experiment and its conclusions honest.

For every position, the dense PyTorch reference runs each candidate response through the model, sums the response-token log probabilities, and normalizes those sequence scores into a student policy. Training blends KL divergence from the Stockfish policy with ordinary loss on Stockfish's top move. The control profile uses the same 900 training positions, 100 validation positions, five epochs, LoRA configuration, optimizer batch, and seed as Phase 5. Packing is disabled because each record is now a group of alternative responses to one prompt.

Run the dense control on the CUDA machine:

```powershell
chessmaxx-train `
  --profile configs/train/scaled-distill-qwen3-0.6b.toml `
  --dataset data/processed/scaled-sft-v1.jsonl `
  --output-dir artifacts/training/scaled-distill-qwen3-0.6b
```

The training report adds candidate count, teacher top-1 probability, teacher entropy, maximum candidate length, and estimated BF16-logit and FP32-loss-tensor bytes for the longest physical batch. Together with measured peak VRAM and throughput, these fields define the dense-memory baseline that a future sparse PyTorch or Triton implementation must beat.

First evaluate normal greedy generation. This remains the authoritative measurement of whether the model can produce a legal move without help:

```powershell
chessmaxx-eval adapter `
  --training-profile configs/train/scaled-distill-qwen3-0.6b.toml `
  --adapter-dir artifacts/training/scaled-distill-qwen3-0.6b/final `
  --dataset data/processed/scaled-sft-v1.jsonl `
  --split validation `
  --selection greedy `
  --stockfish C:\path\to\stockfish.exe `
  --cache artifacts/stockfish-cache.json `
  --output artifacts/evals/scaled-distill-validation-greedy.json
```

Then separately diagnose the learned policy by asking the model to rank every legal move by complete response-sequence likelihood:

```powershell
chessmaxx-eval adapter `
  --training-profile configs/train/scaled-distill-qwen3-0.6b.toml `
  --adapter-dir artifacts/training/scaled-distill-qwen3-0.6b/final `
  --dataset data/processed/scaled-sft-v1.jsonl `
  --split validation `
  --selection legal-rerank `
  --candidate-batch-size 16 `
  --stockfish C:\path\to\stockfish.exe `
  --cache artifacts/stockfish-cache.json `
  --output artifacts/evals/scaled-distill-validation-rerank.json
```

Legal reranking has a structurally 100% legal-move rate, so that number must never be compared with greedy first-try legality. Its useful measurements are Stockfish top-1/top-k agreement, centipawn regret, blunder rate, latency, and candidate-scoring telemetry.

### Phase 6 acceptance checks

On the RTX 3060 Ti, the dense run should:

- Complete without an out-of-memory failure and report peak allocated and reserved VRAM.
- Record finite training and validation policy loss throughout the run.
- Retain the same dataset and profile fingerprints across reports.
- Compare greedy hard-SFT and policy-distilled adapters on the same validation split.
- Select the final checkpoint using validation only, then evaluate it once on the untouched test split.
- Run legal reranking as a separate policy diagnostic, never as a replacement for greedy legality.

The first systems comparison is dense hard SFT versus dense multi-PV policy distillation: throughput, peak VRAM, and chess metrics under the same hardware and data budget. The dense policy implementation is intentionally straightforward; its measured allocation is the target for the later sparse loss and Triton-kernel phases.

## Chunked exact policy loss

Phase 7 keeps the Phase 6 teacher policy, dataset order, LoRA configuration, optimizer batch, seed, and five-epoch budget fixed. It changes only how PyTorch computes target-token probabilities. This isolates an infrastructure question: can exact multi-PV training avoid Qwen's full sequence-by-vocabulary loss tensor?

For a supervised token with hidden state `h`, vocabulary projection `W`, and target token `y`, both implementations calculate:

```text
log p(y | h) = h · W[y] + b[y] - logsumexp(h · W[v] + b[v] for every token v)
```

The dense control materializes logits for every prompt and response position. The chunked implementation instead:

1. Runs Qwen's transformer body without its eager language-model head.
2. Gathers hidden states only at supervised response-token positions.
3. Streams the frozen vocabulary projection in blocks of 4,096 tokens.
4. Combines block normalizers with `logaddexp`, preserving the exact full-softmax denominator.
5. Recomputes one block at a time during backward instead of retaining vocabulary logits.

This is sparse in supervised positions and bounded in vocabulary-memory use, but it is not sampled softmax: every vocabulary row still contributes to the result. The extra backward computation is an intentional memory-for-compute trade. LoRA keeps the output projection frozen while gradients flow through response hidden states into the transformer adapters.

Run the controlled chunked experiment on the CUDA machine:

```powershell
chessmaxx-train `
  --profile configs/train/scaled-distill-qwen3-0.6b-chunked.toml `
  --dataset data/processed/scaled-sft-v1.jsonl `
  --output-dir artifacts/training/scaled-distill-qwen3-0.6b-chunked
```

The dense and chunked profiles must differ only in `name` and `policy_loss_backend`. Their reports include dense full-logit bytes, maximum supervised tokens, per-chunk BF16 and FP32 logit bytes, saved hidden-state bytes, and a theoretical logit-reduction ratio alongside measured peak VRAM and wall time.

### Phase 7 acceptance checks

- Dense and chunked unit tests match target log probabilities, candidate sequence scores, final policy loss, and hidden-state gradients.
- The output projection remains frozen and dense causal-LM logits are never created on the chunked path.
- Training and validation losses remain finite for all 565 optimizer steps.
- Dataset, teacher-policy, LoRA, optimizer, and schedule settings match the Phase 6 control.
- Peak allocated VRAM decreases on the RTX 3060 Ti.
- Wall time, input tokens/second, supervised tokens/second, and positions/hour expose the recomputation cost.
- Validation evaluation checks for a catastrophic capability regression, but the already-consumed Phase 6 test split is not reused for tuning or selection.

Peak reserved memory remains raw allocator telemetry and may exceed physically resident VRAM under the Windows CUDA stack. Peak allocated memory is the in-process comparison until external NVML sampling is added. The sparse PyTorch result becomes the correctness and systems baseline for a later fused Triton implementation.

### Measured RTX 3060 Ti results

All three five-epoch runs used the same 900 training positions, 100 validation positions, effective optimizer batch of eight, Qwen revision, teacher policy, and LoRA configuration.

| Training path | Physical batch | Accumulation | Peak allocated VRAM | Wall time | Positions/hour | Final validation loss |
|---|---:|---:|---:|---:|---:|---:|
| Dense Phase 6 | 2 | 4 | 5.04 GiB | 875.02 s | 18,513.95 | 0.86991 |
| Chunked exact | 2 | 4 | 1.38 GiB | 904.22 s | 17,915.97 | 0.86996 |
| Chunked exact, scaled batch | 8 | 1 | 1.63 GiB | 589.30 s | 27,490.14 | 0.86672 |

The exact chunked loss reduced peak allocated VRAM by 72.66%. That headroom supported a four-times-larger physical batch while preserving the optimizer batch, improving useful chess-position throughput by 53.4% over chunked batch two and 48.5% over dense training. The batch-eight run completed 34.8% faster than chunked batch two without a fatal OOM or a meaningful validation regression. Raw Trainer loss is accumulation-dependent in this Transformers environment; accumulation-normalized losses were 0.81085, 0.80889, and 0.81456 respectively.

## Elo tournament evaluation

Phase 8 plays complete, resumable games instead of scoring frozen positions. Every opening is played twice against each opponent with model colors reversed. Completed games are fsynced to an append-only journal, raw move attempts are retained, and final games are exported as PGN.

Two modes answer different questions:

- `greedy` measures the standalone language model. Its first generated move must parse and be legal; otherwise the game ends in an immediate forfeit.
- `legal-rerank` scores every legal move and chooses the model's highest-likelihood response. This supports complete games and measures the learned policy, but its legality is structural and its rating is always labeled constrained.

Random and material-greedy opponents establish lower-rung capability but have no declared Elo and are excluded from absolute rating. Stockfish opponents may declare a UCI Elo anchor only when the engine identity, `UCI_LimitStrength`, `UCI_Elo`, move time, threads, and hash settings are recorded. The resulting estimate is specific to that fixed Chessmaxx ladder. A clean sweep is reported as `below_ladder` or `above_ladder` with a one-sided 95% bound rather than a fabricated point rating.

Run the four-game wiring check first:

```powershell
chessmaxx-elo `
  --profile configs/elo/qwen3-0.6b-elo-smoke.toml `
  --adapter-dir artifacts/training/scaled-distill-qwen3-0.6b-chunked-batch8/checkpoints/checkpoint-350 `
  --openings data/elo/openings-v1.jsonl `
  --report artifacts/elo/smoke-report.json `
  --journal artifacts/elo/smoke-games.jsonl `
  --pgn artifacts/elo/smoke-games.pgn
```

Then run the constrained 60-game ladder:

```powershell
chessmaxx-elo `
  --profile configs/elo/qwen3-0.6b-elo.toml `
  --adapter-dir artifacts/training/scaled-distill-qwen3-0.6b-chunked-batch8/checkpoints/checkpoint-350 `
  --openings data/elo/openings-v1.jsonl `
  --stockfish C:\path\to\stockfish.exe `
  --report artifacts/elo/constrained-report.json `
  --journal artifacts/elo/constrained-games.jsonl `
  --pgn artifacts/elo/constrained-games.pgn
```

Run the standalone-LLM comparison with the same profile and separate artifacts by adding `--selection greedy`. Never reuse a journal between modes. Reports include win/draw/loss and score rate overall, by opponent, and by model color; selected-move legality; genuine first-attempt legality where applicable; illegal forfeits; termination counts; mean game length and model latency; player identities; adapter/profile/opening fingerprints; generator telemetry; and calibrated Elo with uncertainty when rated games exist.

The first 60-game RTX 3060 Ti run established the Phase 8 wiring result. Greedy generation went 0-0-60 with 4.76% first-attempt legality. Legal reranking completed every game and went 0-36-24 overall: 0-19-1 against deterministic random, 0-17-3 against material-greedy, and 0-0-20 against Stockfish 1320. Because the model scored zero points in its only rated games, the harness correctly reported `below_ladder` with a 95% upper bound of 1033.39 rather than claiming a 1033 rating. The run emitted two recoverable allocator warnings at candidate batch 16; checked-in full profiles now use candidate batch 8.

## Assisted-play evaluation

Phase 9 separates four levels of chess scaffolding:

| Selection | Board reminder | Illegal-move behavior | Legal choices exposed |
|---|---|---|---|
| `greedy` | Fresh FEN every turn | Immediate forfeit | No |
| `retry` | Fresh FEN on every attempt | Explain rejection, up to `max_attempts` | No |
| `retry-with-legal-list` | Fresh FEN on every attempt | Explain rejection, up to `max_attempts` | After the first failure |
| `legal-rerank` | Fresh FEN every turn | Not applicable | Harness scores all legal moves |

Context is an independent variable. `fen` preserves the canonical position-only prompt. `fen-pgn` also supplies SAN history played after the frozen opening; PGN export alone is never treated as model memory. Every generated attempt retains its raw output, parse error, legality, latency, and token counts. Reports distinguish selected-move legality from first-attempt legality, eventual legality, correction success, attempts per move, and assistance overhead. Reranking leaves generation-only metrics undefined rather than reporting a misleading 100% first-try rate.

The tournament accepts either a trained adapter or the pinned base model, enabling an exact control on the same openings, colors, opponents, and seeds:

```powershell
# Base Qwen retry condition
chessmaxx-elo `
  --profile configs/elo/qwen3-0.6b-assisted.toml `
  --base-model-only `
  --selection retry `
  --openings data/elo/openings-v1.jsonl `
  --stockfish C:\path\to\stockfish.exe `
  --report artifacts/elo/base-retry.json `
  --journal artifacts/elo/base-retry.jsonl `
  --pgn artifacts/elo/base-retry.pgn

# Trained Qwen under the identical condition
chessmaxx-elo `
  --profile configs/elo/qwen3-0.6b-assisted.toml `
  --adapter-dir artifacts/training/scaled-distill-qwen3-0.6b-chunked-batch8/checkpoints/checkpoint-350 `
  --selection retry `
  --openings data/elo/openings-v1.jsonl `
  --stockfish C:\path\to\stockfish.exe `
  --report artifacts/elo/adapter-retry.json `
  --journal artifacts/elo/adapter-retry.jsonl `
  --pgn artifacts/elo/adapter-retry.pgn

chessmaxx-elo-compare `
  --reports artifacts/elo/base-retry.json artifacts/elo/adapter-retry.json `
  --output artifacts/elo/base-vs-adapter.json
```

Use `--context fen-pgn` for the memory ablation and separate artifacts for every condition. `chessmaxx-elo-compare` verifies matching schedules and opponent metadata before calculating adapter-minus-base deltas.

The positional test split used in Phase 6 is not reused for tournament tuning. Game results may select neither a new checkpoint nor new hyperparameters; future tuning requires a newly frozen evaluation ladder.

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
