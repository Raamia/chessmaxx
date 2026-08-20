"""Single-game state machine with strict move and draw adjudication."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import chess.pgn

from chessmaxx.evaluation.model import GeneratedMove
from chessmaxx.evaluation.moves import check_generated_move
from chessmaxx.tournament.schema import (
    GameResult,
    MoveAttempt,
    MoveRecord,
    ScheduledGame,
)


_TERMINATIONS = {
    chess.Termination.CHECKMATE: "checkmate",
    chess.Termination.STALEMATE: "stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
    chess.Termination.SEVENTYFIVE_MOVES: "seventyfive_moves",
    chess.Termination.FIVEFOLD_REPETITION: "fivefold_repetition",
}


@dataclass(slots=True)
class ActiveGame:
    schedule: ScheduledGame
    max_plies: int = 300
    assisted_player_id: str | None = None
    max_attempts: int = 1
    board: chess.Board = field(init=False)
    moves: list[MoveRecord] = field(default_factory=list, init=False)
    pending_attempts: list[MoveAttempt] = field(default_factory=list, init=False)
    result: GameResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.max_plies <= 0 or self.max_attempts <= 0:
            raise ValueError("max_plies and max_attempts must be positive")
        if self.max_attempts > 1 and self.assisted_player_id not in {
            self.schedule.white_id,
            self.schedule.black_id,
        }:
            raise ValueError("retrying requires an assisted game participant")
        self.board = chess.Board(self.schedule.initial_fen)

    @property
    def current_player_id(self) -> str:
        return (
            self.schedule.white_id
            if self.board.turn == chess.WHITE
            else self.schedule.black_id
        )

    def apply(self, response: GeneratedMove) -> GameResult | None:
        if self.result is not None:
            raise RuntimeError("cannot move after a game has finished")
        player_id = self.current_player_id
        fen_before = self.board.fen()
        checked = check_generated_move(fen_before, response.raw_output)
        attempt = MoveAttempt(
            attempt=len(self.pending_attempts) + 1,
            raw_output=response.raw_output,
            move_uci=checked.parsed_move,
            legal=checked.is_legal,
            error=checked.error,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            output_tokens=response.output_tokens,
        )
        self.pending_attempts.append(attempt)
        attempt_limit = (
            self.max_attempts if player_id == self.assisted_player_id else 1
        )
        if not checked.is_legal and len(self.pending_attempts) < attempt_limit:
            return None
        record = MoveRecord(
            ply=len(self.moves),
            fen_before=fen_before,
            player_id=player_id,
            raw_output=response.raw_output,
            move_uci=checked.parsed_move,
            legal=checked.is_legal,
            latency_ms=sum(item.latency_ms for item in self.pending_attempts),
            attempts=tuple(self.pending_attempts),
        )
        self.moves.append(record)
        self.pending_attempts.clear()
        if not checked.is_legal or checked.parsed_move is None:
            self.result = self._finish(
                result="0-1" if self.board.turn == chess.WHITE else "1-0",
                termination="illegal_move",
            )
            return self.result

        self.board.push_uci(checked.parsed_move)
        outcome = self.board.outcome(claim_draw=False)
        if outcome is not None:
            termination = _TERMINATIONS.get(outcome.termination)
            if termination is None:
                raise RuntimeError(
                    f"unsupported chess termination: {outcome.termination}"
                )
            self.result = self._finish(
                result=outcome.result(), termination=termination
            )
        elif len(self.moves) >= self.max_plies:
            self.result = self._finish(
                result="1/2-1/2", termination="max_plies"
            )
        return self.result

    def _finish(self, *, result: str, termination: str) -> GameResult:
        return GameResult(
            game_id=self.schedule.game_id,
            opening_id=self.schedule.opening_id,
            initial_fen=self.schedule.initial_fen,
            white_id=self.schedule.white_id,
            black_id=self.schedule.black_id,
            result=result,
            termination=termination,
            final_fen=self.board.fen(),
            moves=tuple(self.moves),
        )


def result_to_pgn(result: GameResult) -> str:
    """Render legal play plus attempted-illegal metadata as one PGN game."""

    game = chess.pgn.Game()
    game.setup(chess.Board(result.initial_fen))
    game.headers.update(
        {
            "Event": "Chessmaxx Elo Evaluation",
            "White": result.white_id,
            "Black": result.black_id,
            "Result": result.result,
            "Termination": result.termination,
            "OpeningId": result.opening_id,
            "GameId": result.game_id,
        }
    )
    node = game
    board = chess.Board(result.initial_fen)
    for record in result.moves:
        if not record.legal or record.move_uci is None:
            game.headers["IllegalPlayer"] = record.player_id
            game.headers["IllegalOutput"] = record.raw_output.replace("\n", " ")
            break
        move = chess.Move.from_uci(record.move_uci)
        node = node.add_variation(move)
        board.push(move)
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter) + "\n"
