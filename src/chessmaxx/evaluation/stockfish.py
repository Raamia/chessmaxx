"""Deterministic Stockfish analysis with a persistent result cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine

from chessmaxx.evaluation.schema import TeacherMove


@dataclass(frozen=True, slots=True)
class StockfishConfig:
    """Engine settings that affect teacher labels and evaluation scores."""

    nodes: int = 50_000
    multipv: int = 3
    threads: int = 1
    hash_mb: int = 64
    mate_score: int = 100_000

    def __post_init__(self) -> None:
        for name in ("nodes", "multipv", "threads", "hash_mb", "mate_score"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class AnalysisCache:
    """Small JSON cache keyed by position, engine identity, and configuration."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._values: dict[str, list[dict[str, Any]]] = {}
        if self.path is not None and self.path.exists():
            self._values = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def key(
        fen: str,
        engine_id: dict[str, str],
        config: StockfishConfig,
        root_move: str | None = None,
    ) -> str:
        payload = json.dumps(
            {
                "fen": fen,
                "engine": engine_id,
                "config": asdict(config),
                "root_move": root_move,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> tuple[TeacherMove, ...] | None:
        value = self._values.get(key)
        if value is None:
            return None
        return tuple(TeacherMove.from_dict(move) for move in value)

    def put(self, key: str, moves: tuple[TeacherMove, ...]) -> None:
        self._values[key] = [move.to_dict() for move in moves]
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._values, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


class StockfishAnalyzer:
    """Analyze boards through a configured UCI engine."""

    def __init__(
        self,
        engine: chess.engine.SimpleEngine,
        config: StockfishConfig,
        cache: AnalysisCache | None = None,
    ) -> None:
        self.engine = engine
        self.config = config
        self.cache = cache or AnalysisCache(None)
        self.engine_id = {str(key): str(value) for key, value in engine.id.items()}

    @classmethod
    def open(
        cls,
        path: str | Path,
        config: StockfishConfig | None = None,
        cache_path: str | Path | None = None,
    ) -> "StockfishAnalyzer":
        selected_config = config or StockfishConfig()
        engine = chess.engine.SimpleEngine.popen_uci(str(path))
        engine.configure(
            {"Threads": selected_config.threads, "Hash": selected_config.hash_mb}
        )
        return cls(engine, selected_config, AnalysisCache(cache_path))

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self) -> "StockfishAnalyzer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def analyze_fen(self, fen: str) -> tuple[TeacherMove, ...]:
        board = chess.Board(fen)
        cache_key = self.cache.key(fen, self.engine_id, self.config)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        result = self.engine.analyse(
            board,
            chess.engine.Limit(nodes=self.config.nodes),
            multipv=self.config.multipv,
            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
        )
        infos = result if isinstance(result, list) else [result]
        moves = self._teacher_moves(board, infos)
        if not moves:
            raise RuntimeError("Stockfish returned no scored principal variations")
        self.cache.put(cache_key, moves)
        return moves

    def score_move(self, fen: str, move_uci: str) -> int:
        """Evaluate one legal root move with the same node budget as MultiPV."""

        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"{move_uci!r} is not legal in the supplied position")
        cache_key = self.cache.key(fen, self.engine_id, self.config, move_uci)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached[0].score_cp

        info = self.engine.analyse(
            board,
            chess.engine.Limit(nodes=self.config.nodes),
            root_moves=[move],
            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
        )
        infos = info if isinstance(info, list) else [info]
        scored = self._teacher_moves(board, infos)
        if not scored:
            raise RuntimeError(f"Stockfish returned no score for {move_uci}")
        self.cache.put(cache_key, scored)
        return scored[0].score_cp

    def _teacher_moves(
        self, board: chess.Board, infos: list[dict[str, Any]]
    ) -> tuple[TeacherMove, ...]:
        moves: list[TeacherMove] = []
        seen: set[chess.Move] = set()
        for info in infos:
            pv = info.get("pv") or []
            score = info.get("score")
            if not pv or score is None or pv[0] in seen or pv[0] not in board.legal_moves:
                continue
            centipawns = score.pov(board.turn).score(mate_score=self.config.mate_score)
            if centipawns is None:
                continue
            seen.add(pv[0])
            moves.append(TeacherMove(move=pv[0].uci(), score_cp=centipawns))
        return tuple(moves)
