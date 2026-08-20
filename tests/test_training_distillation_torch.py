import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from chessmaxx.training.distillation import (  # noqa: E402
    candidate_sequence_log_likelihoods,
    dense_policy_objective,
)
from chessmaxx.training.config import TinySFTProfile  # noqa: E402
from chessmaxx.training.train import _policy_trainer_class  # noqa: E402
from chessmaxx.training.sparse_loss import (  # noqa: E402
    causal_hidden_and_projection,
    chunked_target_log_probabilities,
)
from chessmaxx.evaluation.model import HuggingFaceLegalMoveRanker  # noqa: E402
import chess  # noqa: E402


def test_candidate_scores_sum_only_supervised_next_tokens():
    logits = torch.zeros((1, 2, 4, 5))
    labels = torch.tensor(
        [[[-100, -100, 2, 3], [-100, -100, -100, 4]]]
    )

    scores, counts = candidate_sequence_log_likelihoods(logits, labels)

    assert counts.tolist() == [[2, 1]]
    assert scores.tolist()[0] == pytest.approx(
        [-2 * math.log(5), -math.log(5)]
    )


def test_policy_kl_is_zero_when_student_matches_teacher():
    teacher = torch.tensor([[0.7, 0.3]])
    result = dense_policy_objective(
        teacher.log(),
        torch.ones((1, 2)),
        teacher,
        torch.tensor([[True, True]]),
        hard_loss_weight=0.0,
        student_temperature=1.0,
    )

    assert result["policy_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert result["loss"].item() == pytest.approx(0.0, abs=1e-6)


def test_policy_objective_ignores_padded_candidates():
    result = dense_policy_objective(
        torch.tensor([[-0.2, -0.5, -100.0]]),
        torch.tensor([[2, 2, 0]]),
        torch.tensor([[0.6, 0.4, 0.0]]),
        torch.tensor([[True, True, False]]),
        hard_loss_weight=0.5,
        student_temperature=1.0,
    )

    assert torch.isfinite(result["loss"])
    assert result["student_log_policy"][0, 2].item() == float("-inf")


def test_policy_trainer_backpropagates_through_candidate_logits():
    class BaseTrainer:
        pass

    logits = torch.randn((2, 4, 7), requires_grad=True)

    class Model:
        def __call__(self, **kwargs):
            assert kwargs["input_ids"].shape == (2, 4)
            return SimpleNamespace(logits=logits)

    profile = TinySFTProfile(
        name="policy",
        model_id="model",
        objective="multipv_policy",
        packing=False,
    )
    trainer_type = _policy_trainer_class(
        SimpleNamespace(Trainer=BaseTrainer), profile
    )
    trainer = trainer_type()
    loss = trainer.compute_loss(
        Model(),
        {
            "input_ids": torch.ones((1, 2, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 2, 4), dtype=torch.long),
            "labels": torch.tensor(
                [[[-100, -100, 2, 3], [-100, -100, 4, 5]]]
            ),
            "teacher_probabilities": torch.tensor([[0.6, 0.4]]),
            "candidate_mask": torch.tensor([[True, True]]),
        },
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_legal_move_ranker_selects_highest_likelihood_sequence():
    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 3
        pad_token_id = 0
        padding_side = "right"

        def __init__(self):
            self.move_tokens = {}

        def encode(self, text, *, add_special_tokens):
            if text.startswith(" "):
                return [self.move_tokens.setdefault(text.strip(), 10 + len(self.move_tokens))]
            return [1, 2]

        def __len__(self):
            return 64

    class Model:
        config = SimpleNamespace(
            _commit_hash="revision",
            vocab_size=64,
            hidden_size=8,
            num_hidden_layers=1,
        )
        dtype = torch.float32

        def eval(self):
            return self

        def parameters(self):
            return []

        def __call__(self, *, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            logits = torch.zeros((*input_ids.shape, 64))
            for row in range(input_ids.shape[0]):
                move_token = int(input_ids[row, 2])
                logits[row, 1, move_token] = (move_token - 10) / 10
                logits[row, 2, 3] = 20.0
            return SimpleNamespace(logits=logits)

    tokenizer = Tokenizer()
    ranker = HuggingFaceLegalMoveRanker(
        Model(), tokenizer, "fake", "cpu", candidate_batch_size=4
    )

    result = ranker.generate_many([chess.STARTING_FEN])[0]

    assert result.raw_output == sorted(
        move.uci() for move in chess.Board().legal_moves
    )[-1]
    assert ranker.telemetry["candidate_sequences_scored"] == 20


def test_extracts_hidden_states_without_running_the_dense_lm_head():
    expected_hidden = torch.randn((2, 4, 3))
    output_head = torch.nn.Linear(3, 7, bias=False)

    class Backbone:
        def __call__(self, **kwargs):
            assert kwargs["input_ids"].shape == (2, 4)
            assert kwargs["attention_mask"].shape == (2, 4)
            assert kwargs["use_cache"] is False
            assert kwargs["return_dict"] is True
            return SimpleNamespace(last_hidden_state=expected_hidden)

    class CausalLM:
        model = Backbone()

        def get_output_embeddings(self):
            return output_head

        def __call__(self, **kwargs):
            raise AssertionError("the dense causal-LM forward must not run")

    class PeftWrapper:
        def get_base_model(self):
            return CausalLM()

    projection = causal_hidden_and_projection(
        PeftWrapper(),
        input_ids=torch.ones((2, 4), dtype=torch.long),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
    )

    assert projection.hidden_states is expected_hidden
    assert projection.weight is output_head.weight
    assert projection.bias is None


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 16])
def test_chunked_target_log_probabilities_match_dense_softmax(chunk_size):
    torch.manual_seed(7)
    hidden = torch.randn((5, 4))
    weight = torch.randn((7, 4))
    bias = torch.randn(7)
    targets = torch.tensor([0, 6, 2, 4, 1])
    expected = torch.nn.functional.log_softmax(
        torch.nn.functional.linear(hidden, weight, bias).float(), dim=-1
    ).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    actual = chunked_target_log_probabilities(
        hidden,
        weight,
        targets,
        bias=bias,
        chunk_size=chunk_size,
    )

    assert actual.dtype == torch.float32
    assert actual.tolist() == pytest.approx(expected.tolist(), abs=1e-6)
