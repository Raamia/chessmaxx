"""Deterministic, game-aware sampling for frozen PGN evaluation sets."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.pgn

from chessmaxx.evaluation.schema import EvaluationPosition


PHASES = ("opening", "middlegame", "endgame")
PIECE_VALUES = {
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def classify_phase(board: chess.Board, ply: int) -> str:
    """Classify a board using fixed, inspectable material heuristics."""

    non_pawn_material = sum(
        len(board.pieces(piece_type, color)) * value
        for color in chess.COLORS
        for piece_type, value in PIECE_VALUES.items()
    )
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    if queens == 0 or non_pawn_material <= 2_600:
        return "endgame"
    if ply < 20:
        return "opening"
    return "middlegame"


def _rank(seed: int, position_id: str, fen: str) -> int:
    digest = hashlib.sha256(f"{seed}:{position_id}:{fen}".encode()).digest()
    return int.from_bytes(digest, "big")


def _game_id(game: chess.pgn.Game, game_index: int, moves: list[chess.Move]) -> str:
    signature = json.dumps(
        {
            "headers": sorted(game.headers.items()),
            "moves": [move.uci() for move in moves],
        },
        separators=(",", ":"),
    )
    digest = hashlib.sha256(signature.encode()).hexdigest()[:12]
    return f"pgn-{game_index:08d}-{digest}"


def _read_games(path: Path) -> Iterator[tuple[int, chess.pgn.Game]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        game_index = 0
        while game := chess.pgn.read_game(handle):
            game_index += 1
            if game.errors:
                raise ValueError(f"{path}: game {game_index} contains PGN errors")
            yield game_index, game


def sample_pgn_positions(
    path: str | Path,
    count: int,
    *,
    seed: int = 0,
    minimum_ply: int = 8,
    max_per_game: int = 4,
) -> list[EvaluationPosition]:
    """Select a phase-balanced sample without retaining an entire PGN database."""

    if count <= 0:
        raise ValueError("count must be positive")
    if minimum_ply < 0:
        raise ValueError("minimum_ply must be non-negative")
    if max_per_game <= 0:
        raise ValueError("max_per_game must be positive")

    source = Path(path)
    buckets: dict[str, list[tuple[int, str, EvaluationPosition]]] = {
        phase: [] for phase in PHASES
    }
    for game_index, game in _read_games(source):
        moves = list(game.mainline_moves())
        game_id = _game_id(game, game_index, moves)
        board = game.board()
        game_candidates: list[tuple[int, str, EvaluationPosition]] = []
        for ply, move in enumerate(moves):
            if ply >= minimum_ply and not board.is_game_over():
                fen = board.fen()
                position_id = f"{game_id}-ply-{ply:03d}"
                position = EvaluationPosition(
                    position_id=position_id,
                    fen=fen,
                    game_id=game_id,
                    ply=ply,
                    phase=classify_phase(board, ply),
                    metadata={
                        "source": source.name,
                        "source_game_index": game_index,
                    },
                )
                rank = _rank(seed, position_id, fen)
                game_candidates.append((rank, position_id, position))
            board.push(move)

        for rank, position_id, position in sorted(game_candidates)[:max_per_game]:
            bucket = buckets[position.phase or "middlegame"]
            item = (-rank, position_id, position)
            if len(bucket) < count:
                heapq.heappush(bucket, item)
            elif item > bucket[0]:
                heapq.heapreplace(bucket, item)

    candidates = {
        phase: sorted(
            [(-negative_rank, position_id, position) for negative_rank, position_id, position in heap]
        )
        for phase, heap in buckets.items()
    }
    base, remainder = divmod(count, len(PHASES))
    quotas = {
        phase: base + (index < remainder) for index, phase in enumerate(PHASES)
    }
    selected: list[tuple[int, str, EvaluationPosition]] = []
    leftovers: list[tuple[int, str, EvaluationPosition]] = []
    for phase in PHASES:
        quota = quotas[phase]
        selected.extend(candidates[phase][:quota])
        leftovers.extend(candidates[phase][quota:])
    if len(selected) < count:
        selected.extend(sorted(leftovers)[: count - len(selected)])
    if not selected:
        raise ValueError(f"{source}: no eligible positions found")
    return [position for _, _, position in sorted(selected)]
