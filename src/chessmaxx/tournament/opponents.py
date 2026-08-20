"""Deterministic local opponents for an auditable rating ladder."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine

from chessmaxx.evaluation.model import GeneratedMove


class DeterministicLegalMoveGenerator:
    def __init__(self, player_id: str, *, kind: str, seed: int = 2026) -> None:
        self.player_id = player_id
        self.kind = kind
        self.seed = seed
        self.reset_telemetry()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "kind": self.kind,
            "seed": self.seed,
        }

    @property
    def telemetry(self) -> dict[str, Any]:
        return {"positions_generated": self._positions_generated}

    def reset_telemetry(self) -> None:
        self._positions_generated = 0

    def generate_many(self, fens: list[str]) -> list[GeneratedMove]:
        results = [self._generate(fen) for fen in fens]
        self._positions_generated += len(results)
        return results

    def _generate(self, fen: str) -> GeneratedMove:
        started = time.perf_counter()
        board = chess.Board(fen)
        moves = sorted(move.uci() for move in board.legal_moves)
        if not moves:
            raise ValueError("opponent received a terminal position")
        selected = self._select(board, moves)
        return GeneratedMove(
            raw_output=selected,
            latency_ms=(time.perf_counter() - started) * 1000,
            output_tokens=None,
            prompt_tokens=None,
        )

    def _select(self, board: chess.Board, moves: list[str]) -> str:
        if self.kind == "random":
            digest = hashlib.sha256(
                f"{self.seed}\0{board.fen()}".encode()
            ).digest()
            return moves[int.from_bytes(digest[:8], "big") % len(moves)]
        if self.kind == "material":
            return max(moves, key=lambda move: (self._material_score(board, move), move))
        raise ValueError(f"unsupported deterministic opponent kind: {self.kind}")

    @staticmethod
    def _material_score(board: chess.Board, move_uci: str) -> int:
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }
        mover = board.turn
        child = board.copy(stack=False)
        child.push_uci(move_uci)
        if child.is_checkmate():
            return 1_000_000
        balance = sum(
            len(child.pieces(piece_type, chess.WHITE)) * value
            - len(child.pieces(piece_type, chess.BLACK)) * value
            for piece_type, value in values.items()
        )
        return balance if mover == chess.WHITE else -balance


class StockfishMoveGenerator:
    def __init__(
        self,
        engine: chess.engine.SimpleEngine,
        *,
        player_id: str,
        rating: float | None,
        move_time_ms: int,
        settings: dict[str, Any],
    ) -> None:
        if move_time_ms <= 0:
            raise ValueError("move_time_ms must be positive")
        self.engine = engine
        self.player_id = player_id
        self.rating = rating
        self.move_time_ms = move_time_ms
        self.settings = settings
        self.reset_telemetry()

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        player_id: str,
        rating: float | None,
        move_time_ms: int = 100,
        threads: int = 1,
        hash_mb: int = 64,
        uci_elo: int | None = None,
        skill_level: int | None = None,
    ) -> "StockfishMoveGenerator":
        engine = chess.engine.SimpleEngine.popen_uci(str(path))
        settings: dict[str, Any] = {"Threads": threads, "Hash": hash_mb}
        if uci_elo is not None:
            settings.update({"UCI_LimitStrength": True, "UCI_Elo": uci_elo})
        if skill_level is not None:
            settings["Skill Level"] = skill_level
        unsupported = set(settings) - set(engine.options)
        if unsupported:
            engine.quit()
            raise ValueError(
                f"Stockfish does not support option(s): {', '.join(sorted(unsupported))}"
            )
        engine.configure(settings)
        return cls(
            engine,
            player_id=player_id,
            rating=rating,
            move_time_ms=move_time_ms,
            settings=settings,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "kind": "stockfish",
            "rating": self.rating,
            "engine": {str(key): str(value) for key, value in self.engine.id.items()},
            "move_time_ms": self.move_time_ms,
            "settings": dict(self.settings),
        }

    @property
    def telemetry(self) -> dict[str, Any]:
        return {
            "positions_generated": self._positions_generated,
            "generation_seconds": self._generation_seconds,
        }

    def reset_telemetry(self) -> None:
        self._positions_generated = 0
        self._generation_seconds = 0.0

    def generate_many(self, fens: list[str]) -> list[GeneratedMove]:
        responses: list[GeneratedMove] = []
        for fen in fens:
            board = chess.Board(fen)
            started = time.perf_counter()
            result = self.engine.play(
                board, chess.engine.Limit(time=self.move_time_ms / 1000)
            )
            elapsed = time.perf_counter() - started
            if result.move is None:
                raise RuntimeError("Stockfish returned no move")
            responses.append(
                GeneratedMove(result.move.uci(), latency_ms=elapsed * 1000)
            )
            self._positions_generated += 1
            self._generation_seconds += elapsed
        return responses

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self) -> "StockfishMoveGenerator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
