"""Round-based tournament execution that batches turns across live games."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from chessmaxx.evaluation.model import MoveGenerator
from chessmaxx.tournament.game import ActiveGame
from chessmaxx.tournament.prompts import build_retry_prompt, san_history
from chessmaxx.tournament.schema import GameResult, ScheduledGame


class TournamentRunner:
    def __init__(
        self,
        generators: dict[str, MoveGenerator],
        *,
        batch_size: int = 8,
        max_plies: int = 300,
        assisted_player_id: str | None = None,
        max_attempts: int = 1,
        include_legal_moves: bool = False,
        include_move_history: bool = False,
        on_result: Callable[[GameResult], None] | None = None,
    ) -> None:
        if min(batch_size, max_plies, max_attempts) <= 0:
            raise ValueError("tournament batch and game limits must be positive")
        if not generators:
            raise ValueError("tournament requires at least one move generator")
        if (max_attempts > 1 or include_move_history) and (
            assisted_player_id not in generators
        ):
            raise ValueError("prompt assistance requires a model player generator")
        if include_legal_moves and max_attempts == 1:
            raise ValueError("legal-move feedback requires retry attempts")
        self.generators = generators
        self.batch_size = batch_size
        self.max_plies = max_plies
        self.assisted_player_id = assisted_player_id
        self.max_attempts = max_attempts
        self.include_legal_moves = include_legal_moves
        self.include_move_history = include_move_history
        self.on_result = on_result

    def run(self, schedules: Sequence[ScheduledGame]) -> tuple[GameResult, ...]:
        required_players = {
            player
            for schedule in schedules
            for player in (schedule.white_id, schedule.black_id)
        }
        missing = required_players - set(self.generators)
        if missing:
            raise ValueError(
                f"no move generator for player(s): {', '.join(sorted(missing))}"
            )
        for generator in self.generators.values():
            reset = getattr(generator, "reset_telemetry", None)
            if callable(reset):
                reset()
        active = [
            ActiveGame(
                schedule,
                max_plies=self.max_plies,
                assisted_player_id=self.assisted_player_id,
                max_attempts=self.max_attempts,
            )
            for schedule in schedules
        ]
        results: dict[str, GameResult] = {}
        while len(results) < len(active):
            turns: dict[str, list[ActiveGame]] = defaultdict(list)
            for game in active:
                if game.result is None:
                    turns[game.current_player_id].append(game)
            if not turns:
                raise RuntimeError("tournament made no progress")
            for player_id in sorted(turns):
                generator = self.generators[player_id]
                games = turns[player_id]
                for start in range(0, len(games), self.batch_size):
                    batch = games[start : start + self.batch_size]
                    prompted_turn = player_id == self.assisted_player_id and (
                        self.max_attempts > 1 or self.include_move_history
                    )
                    if prompted_turn:
                        generate_prompts = getattr(generator, "generate_prompts", None)
                        if not callable(generate_prompts):
                            raise TypeError(
                                "assisted player generator must support explicit prompts"
                            )
                        responses = generate_prompts(
                            [
                                build_retry_prompt(
                                    game.board.fen(),
                                    tuple(game.pending_attempts),
                                    include_legal_moves=(
                                        self.include_legal_moves
                                        and bool(game.pending_attempts)
                                    ),
                                    move_history=(
                                        san_history(game.board)
                                        if self.include_move_history
                                        else None
                                    ),
                                )
                                for game in batch
                            ]
                        )
                    else:
                        responses = generator.generate_many(
                            [game.board.fen() for game in batch]
                        )
                    if len(responses) != len(batch):
                        raise RuntimeError(
                            f"generator {player_id!r} returned the wrong batch size"
                        )
                    for game, response in zip(batch, responses, strict=True):
                        result = game.apply(response)
                        if result is not None:
                            results[result.game_id] = result
                            if self.on_result is not None:
                                self.on_result(result)
        return tuple(results[schedule.game_id] for schedule in schedules)

    @property
    def telemetry(self) -> dict[str, Any]:
        return {
            player_id: dict(getattr(generator, "telemetry", {}))
            for player_id, generator in self.generators.items()
        }
