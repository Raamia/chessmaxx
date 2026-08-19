import json

import chess
import chess.engine

from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.evaluation.stockfish import (
    AnalysisCache,
    StockfishAnalyzer,
    StockfishConfig,
)


class FakeEngine:
    id = {"name": "Fakefish 1.0", "author": "tests"}

    def __init__(self):
        self.calls = 0
        self.closed = False

    def analyse(self, board, limit, **kwargs):
        self.calls += 1
        if kwargs.get("root_moves"):
            move = kwargs["root_moves"][0]
            return {
                "pv": [move],
                "score": chess.engine.PovScore(chess.engine.Cp(12), board.turn),
            }
        return [
            {
                "pv": [chess.Move.from_uci("e2e4")],
                "score": chess.engine.PovScore(chess.engine.Cp(30), chess.WHITE),
            },
            {
                "pv": [chess.Move.from_uci("d2d4")],
                "score": chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE),
            },
        ]

    def quit(self):
        self.closed = True


class OutOfOrderEngine(FakeEngine):
    def analyse(self, board, limit, **kwargs):
        self.calls += 1
        return [
            {
                "pv": [chess.Move.from_uci("e2e4")],
                "score": chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE),
            },
            {
                "pv": [chess.Move.from_uci("d2d4")],
                "score": chess.engine.PovScore(chess.engine.Cp(30), chess.WHITE),
            },
            {
                "pv": [chess.Move.from_uci("c2c4")],
                "score": chess.engine.PovScore(chess.engine.Cp(30), chess.WHITE),
            },
        ]


def test_analysis_is_scored_from_side_to_move_and_cached(tmp_path):
    engine = FakeEngine()
    analyzer = StockfishAnalyzer(
        engine,  # type: ignore[arg-type]
        StockfishConfig(nodes=100, multipv=2),
        AnalysisCache(tmp_path / "cache.json"),
    )

    first = analyzer.analyze_fen(chess.STARTING_FEN)
    second = analyzer.analyze_fen(chess.STARTING_FEN)

    assert [(move.move, move.score_cp) for move in first] == [
        ("e2e4", 30),
        ("d2d4", 20),
    ]
    assert second == first
    assert engine.calls == 1


def test_fresh_multipv_is_normalized_and_persisted(tmp_path):
    cache_path = tmp_path / "cache.json"
    analyzer = StockfishAnalyzer(
        OutOfOrderEngine(),  # type: ignore[arg-type]
        StockfishConfig(nodes=100, multipv=3),
        AnalysisCache(cache_path),
    )

    moves = analyzer.analyze_fen(chess.STARTING_FEN)

    assert [(move.move, move.score_cp) for move in moves] == [
        ("c2c4", 30),
        ("d2d4", 30),
        ("e2e4", 20),
    ]
    cache_values = json.loads(cache_path.read_text(encoding="utf-8")).values()
    persisted = next(iter(cache_values))
    assert [(move["move"], move["score_cp"]) for move in persisted] == [
        ("c2c4", 30),
        ("d2d4", 30),
        ("e2e4", 20),
    ]


def test_cached_out_of_order_result_is_normalized_and_rewritten(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "legacy": [
                    TeacherMove(move="e2e4", score_cp=20).to_dict(),
                    TeacherMove(move="d2d4", score_cp=30).to_dict(),
                    TeacherMove(move="c2c4", score_cp=30).to_dict(),
                ]
            }
        ),
        encoding="utf-8",
    )
    cache = AnalysisCache(cache_path)

    moves = cache.get("legacy")

    assert moves is not None
    assert [(move.move, move.score_cp) for move in moves] == [
        ("c2c4", 30),
        ("d2d4", 30),
        ("e2e4", 20),
    ]
    persisted = json.loads(cache_path.read_text(encoding="utf-8"))["legacy"]
    assert [(move["move"], move["score_cp"]) for move in persisted] == [
        ("c2c4", 30),
        ("d2d4", 30),
        ("e2e4", 20),
    ]


def test_analysis_flips_score_for_black_to_move():
    engine = FakeEngine()
    analyzer = StockfishAnalyzer(engine, StockfishConfig())  # type: ignore[arg-type]
    board = chess.Board(chess.STARTING_FEN)
    board.turn = chess.BLACK
    info = {
        "pv": [chess.Move.from_uci("e7e5")],
        "score": chess.engine.PovScore(chess.engine.Cp(40), chess.WHITE),
    }

    moves = analyzer._teacher_moves(board, [info])

    assert moves[0].score_cp == -40


def test_context_manager_closes_engine():
    engine = FakeEngine()

    with StockfishAnalyzer(engine, StockfishConfig()):  # type: ignore[arg-type]
        pass

    assert engine.closed is True


def test_scores_and_caches_a_specific_legal_move(tmp_path):
    engine = FakeEngine()
    analyzer = StockfishAnalyzer(
        engine,  # type: ignore[arg-type]
        StockfishConfig(nodes=100),
        AnalysisCache(tmp_path / "cache.json"),
    )

    assert analyzer.score_move(chess.STARTING_FEN, "g1f3") == 12
    assert analyzer.score_move(chess.STARTING_FEN, "g1f3") == 12
    assert engine.calls == 1
