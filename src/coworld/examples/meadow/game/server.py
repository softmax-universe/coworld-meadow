"""Meadow game server.

Wraps the pure engine in the Coworld game-container contract (see
`docs/roles/GAME.md`): config via `COGAME_CONFIG_URI`, results and replay
written to runner URIs, `/healthz`, browser clients, and the `/player`,
`/global`, and `/admin` websocket routes. Replay viewing is the static
browser bundle declared in the manifest (`static-replay-viewer/`), not a
container route.

Unlike Paint Arena's fixed-rate ticks, Meadow advances on a round barrier: a
round settles as soon as every player has submitted a valid action, or fails the
episode if any player submits an invalid decision or misses `round_seconds`.
No substitute or fallback move is ever fabricated for a missing or invalid seat.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from coworld.examples.meadow.game.engine import (
    MeadowConfig,
    MeadowState,
    episode_results,
    new_state,
    observation,
    step,
)
from coworld.examples.meadow.shared.artifact_io import (
    artifact_method,
    read_data,
    write_data,
)
from coworld.examples.meadow.shared.decision import (
    AttemptProgress,
    PlayerDecision,
    apply_player_decision,
    executed_action,
    parse_reply,
)
from coworld.examples.meadow.shared.log_shipper import get_logger
from coworld.examples.meadow.shared.prompts import PromptConfig
from coworld.examples.meadow.shared.scenarios import scenario_family
from coworld.examples.meadow.shared.trajectory import (
    Attempt,
    DecisionRecord,
    Trajectory,
)

CLIENT_DIR = Path(__file__).parent / "client"
logger = get_logger("meadow.game")
GAME_HOST = os.environ.get("COGAME_HOST", "0.0.0.0")
GAME_PORT = int(os.environ.get("COGAME_PORT", "8080"))
# Keep serving after the final round so viewers (and the hosted certifier's
# websocket probes) can still read the final state: an all-scripted episode can
# finish in under a second, and exiting immediately races anything that
# connected while the game was live.
POST_GAME_LINGER_SECONDS = float(
    os.environ.get("COWORLD_MEADOW_POST_GAME_LINGER_SECONDS", "30")
)
# Hard ceiling on the post-game hold: even with a viewer attached (the hosted
# certifier holds /global open while it waits for player pods, which can take
# a minute on cold nodes), the server eventually exits.
POST_GAME_MAX_LINGER_SECONDS = float(
    os.environ.get("COWORLD_MEADOW_POST_GAME_MAX_LINGER_SECONDS", "90")
)

RAW_CONFIG: dict[str, Any] = json.loads(read_data(os.environ["COGAME_CONFIG_URI"]))
RESULTS_URI = os.environ["COGAME_RESULTS_URI"]
REPLAY_URI = os.environ["COGAME_SAVE_REPLAY_URI"]

TOKENS: list[str] = RAW_CONFIG["tokens"]
PLAYER_NAMES: list[str] = [player["name"] for player in RAW_CONFIG["players"]]
PROMPT_SELECTION = PromptConfig.model_validate(RAW_CONFIG).resolve()
CONFIG = MeadowConfig.model_validate({**RAW_CONFIG, "num_players": len(TOKENS)})
ROUND_SECONDS = float(RAW_CONFIG.get("round_seconds", 20.0))
PLAYER_CONNECT_TIMEOUT_SECONDS = float(
    RAW_CONFIG.get("player_connect_timeout_seconds", 180)
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    timeout_task = (
        asyncio.create_task(_start_after_player_connect_timeout()) if TOKENS else None
    )
    yield
    if timeout_task is not None:
        timeout_task.cancel()
        with suppress(asyncio.CancelledError):
            await timeout_task


app = FastAPI(lifespan=lifespan)
server: uvicorn.Server


class GameSession:
    def __init__(self) -> None:
        self.engine: MeadowState = new_state(CONFIG)
        self.players: dict[int, WebSocket] = {}
        self.pending: dict[int, PlayerDecision] = {}
        self.frames: list[dict[str, Any]] = []
        self.started = False
        self.done = False
        self.failed = False
        self.failure_reason: str | None = None
        self.paused = False
        self.round_seconds = ROUND_SECONDS
        self.global_viewers = 0
        self.progress: dict[tuple[int, int], list[Attempt]] = {}
        self.trajectory = (
            Trajectory(
                episode_id=os.environ["COWORLD_EPISODE_ID"],
                game_version=os.environ["COWORLD_GAME_VERSION"],
                source_revision=os.environ["COWORLD_SOURCE_REVISION"],
                image_digest=os.environ.get("COWORLD_IMAGE_DIGEST"),
                seed_family=scenario_family(CONFIG),
            )
            if os.environ.get("COGAME_SAVE_TRAJECTORY_URI")
            else None
        )

    def capture_attempt(self, slot: int, progress: AttemptProgress) -> None:
        """Preserve known started and late attempts without changing an executed decision."""
        if self.trajectory is not None and self.trajectory.finished:
            return
        attempt = progress.attempt.model_copy(deep=True)
        attempt.accepted = False
        if attempt.origin in {"teacher", "human"}:
            attempt.origin = "unknown"
        if attempt.response is not None:
            parsed = parse_reply(attempt.response, slot, CONFIG)
            attempt.parsed_action = parsed.model_dump() if parsed is not None else None
        attempts = self.progress.setdefault((progress.round, slot), [])
        if self.trajectory is not None:
            for record in self.trajectory.records:
                if (
                    isinstance(record, DecisionRecord)
                    and record.decision_id == f"round-{progress.round}-seat-{slot}"
                ):
                    if record.selected_attempt_id == attempt.attempt_id:
                        return
                    attempts = record.attempts
                    if record.action_status == "failed":
                        attempt.rejection_reason = (
                            "completed after round deadline"
                            if attempt.latency_ms is not None
                            else "native call in progress at round deadline"
                        )
        existing = next(
            (
                index
                for index, item in enumerate(attempts)
                if item.attempt_id == attempt.attempt_id
            ),
            None,
        )
        if existing is None:
            attempts.append(attempt)
        else:
            attempts[existing] = attempt


session = GameSession()


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/client/global")
def global_client() -> HTMLResponse:
    return HTMLResponse((CLIENT_DIR / "global.html").read_text())


@app.get("/client/admin")
def admin_client() -> HTMLResponse:
    return HTMLResponse((CLIENT_DIR / "admin.html").read_text())


@app.get("/client/player")
def player_client() -> HTMLResponse:
    return HTMLResponse((CLIENT_DIR / "player.html").read_text())


@app.websocket("/global")
async def global_viewer(websocket: WebSocket) -> None:
    await websocket.accept()
    session.global_viewers += 1
    try:
        sender = asyncio.create_task(_send_global_snapshots(websocket))
        receiver = asyncio.create_task(_drain_messages(websocket))
        done, pending = await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    finally:
        session.global_viewers -= 1


GLOBAL_KEEPALIVE_SECONDS = 15.0
GLOBAL_MIN_SEND_INTERVAL_SECONDS = 1.0


async def _send_global_snapshots(websocket: WebSocket) -> None:
    # Send only on game progress (round/done/collapse), coalesced to at most
    # one message per second, plus a slow keepalive. The hosted certifier
    # holds this socket WITHOUT reading while it verifies player pods, and its
    # websocket client stops reading the transport — including Pong frames —
    # once ~16 messages sit unread. An unconditional 2Hz stream fills that
    # budget during any pod-start delay and the certification ping then times
    # out against a perfectly healthy server, so the total sent while a viewer
    # isn't reading must stay far below that queue limit.
    loop = asyncio.get_running_loop()
    await websocket.send_json(_snapshot())
    sent_at = loop.time()
    sent_progress = (session.engine.round, session.done, session.engine.collapsed)
    while True:
        await asyncio.sleep(0.5)
        now = loop.time()
        progress = (session.engine.round, session.done, session.engine.collapsed)
        changed = (
            progress != sent_progress
            and now - sent_at >= GLOBAL_MIN_SEND_INTERVAL_SECONDS
        )
        if changed or now - sent_at >= GLOBAL_KEEPALIVE_SECONDS:
            await websocket.send_json(_snapshot())
            sent_at = now
            sent_progress = progress


async def _drain_messages(websocket: WebSocket) -> None:
    async for _ in websocket.iter_json():
        pass


@app.websocket("/admin")
async def admin(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json(_snapshot())
    async for command in websocket.iter_json():
        if command["command"] == "pause":
            session.paused = True
        elif command["command"] == "resume":
            session.paused = False
        elif command["command"] == "round_seconds":
            session.round_seconds = float(command["round_seconds"])
        await websocket.send_json(_snapshot())


@app.websocket("/player")
async def player(websocket: WebSocket) -> None:
    slot = int(websocket.query_params["slot"])
    token = websocket.query_params["token"]
    if slot < 0 or slot >= len(TOKENS) or TOKENS[slot] != token:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    session.players[slot] = websocket
    logger.info(
        "player slot %d connected (%d/%d)", slot, len(session.players), len(TOKENS)
    )
    await websocket.send_json(_player_observation(slot))
    if len(session.players) == len(TOKENS) and not session.started:
        session.started = True
        logger.info("all players connected, starting game")
        asyncio.create_task(_play_game())

    try:
        async for message in websocket.iter_json():
            if isinstance(message, dict) and message.get("type") == "attempt":
                session.capture_attempt(slot, AttemptProgress.model_validate(message))
                continue
            if not session.done and session.engine.round < CONFIG.rounds:
                try:
                    decision = PlayerDecision.model_validate(message)
                except ValidationError:
                    session.pending[slot] = PlayerDecision(
                        round=session.engine.round,
                        action=None,
                        attempts=session.progress.get((session.engine.round, slot), []),
                        failure_reason="invalid-action-envelope",
                    )
                    continue
                if decision.round == session.engine.round:
                    session.pending[slot] = apply_player_decision(
                        decision, slot, CONFIG
                    )
    finally:
        if session.players.get(slot) is websocket:
            del session.players[slot]


async def _start_after_player_connect_timeout() -> None:
    await asyncio.sleep(PLAYER_CONNECT_TIMEOUT_SECONDS)
    if not session.started and not session.done:
        session.started = True
        logger.info("player connect timeout elapsed, starting game")
        asyncio.create_task(_play_game())


async def _write_trajectory() -> None:
    if session.trajectory is None:
        return
    uri = os.environ["COGAME_SAVE_TRAJECTORY_URI"]
    if uri.startswith("file://"):
        from urllib.parse import unquote, urlparse

        await asyncio.to_thread(
            session.trajectory.write, Path(unquote(urlparse(uri).path))
        )
    else:
        body = (
            "\n".join(
                record.model_dump_json() for record in session.trajectory.records
            )
            + "\n"
        )
        await asyncio.to_thread(
            write_data,
            uri,
            body,
            content_type="application/x-ndjson",
            http_method=artifact_method("COGAME_SAVE_TRAJECTORY_METHOD"),
        )


async def _drain_trajectory_attempts(loop: asyncio.AbstractEventLoop) -> None:
    if session.trajectory is None:
        return
    drain_deadline = loop.time() + session.round_seconds
    while loop.time() < drain_deadline and any(
        attempt.latency_ms is None
        for record in session.trajectory.records
        if isinstance(record, DecisionRecord)
        for attempt in record.attempts
    ):
        await asyncio.sleep(0.05)


async def _play_game() -> None:
    loop = asyncio.get_running_loop()
    await asyncio.sleep(0.5)
    while session.engine.round < CONFIG.rounds:
        if session.paused:
            await asyncio.sleep(0.1)
            continue
        deadline = loop.time() + session.round_seconds
        while loop.time() < deadline:
            if any(
                decision.action is None or decision.failure_reason is not None
                for decision in session.pending.values()
            ):
                break
            if len(session.pending) == CONFIG.num_players:
                break
            await asyncio.sleep(0.05)

        decisions = [
            session.pending.get(
                slot,
                PlayerDecision(
                    round=session.engine.round,
                    action=None,
                    attempts=session.progress.get((session.engine.round, slot), []),
                    failure_reason="missing-or-late-player",
                ),
            )
            for slot in range(CONFIG.num_players)
        ]
        views = [_player_observation(slot) for slot in range(CONFIG.num_players)]
        session.pending.clear()
        failed = [
            (slot, decision)
            for slot, decision in enumerate(decisions)
            if decision.action is None or decision.failure_reason is not None
        ]
        if failed:
            summary = "; ".join(
                f"seat {slot}: {decision.failure_reason or 'missing-action'}"
                for slot, decision in failed
            )
            session.failed = True
            session.failure_reason = summary
            logger.error(
                "episode failed at round %d (%s); no substitute action played and no scores published",
                session.engine.round,
                summary,
            )
            if session.trajectory is not None:
                for slot, decision in enumerate(decisions):
                    if decision.action is None or decision.failure_reason is not None:
                        attempts = decision.attempts
                        reason = decision.failure_reason or "missing-action"
                    else:
                        attempts = [a.model_copy(deep=True) for a in decision.attempts]
                        for attempt in attempts:
                            attempt.accepted = False
                            attempt.rejection_reason = (
                                "round aborted due to peer policy failure"
                            )
                        reason = "episode-aborted-by-peer-failure"
                    session.trajectory.record(
                        decision_id=f"round-{session.engine.round}-seat-{slot}",
                        seat=slot,
                        observation=views[slot],
                        prompt=None,
                        attempts=attempts,
                        executed_action=None,
                        failure_reason=reason,
                        terminal=True,
                    )
                await _drain_trajectory_attempts(loop)
                session.trajectory.finish(
                    outcome=None,
                    participant_outcomes=None,
                    completed=False,
                    failure_reason=summary,
                )
                await _write_trajectory()
            await asyncio.to_thread(
                write_data,
                REPLAY_URI,
                json.dumps(
                    {
                        **_replay_payload(None),
                        "failed": True,
                        "failure_reason": summary,
                    }
                ),
                content_type="application/json",
                http_method=artifact_method("COGAME_SAVE_REPLAY_METHOD"),
            )
            player_failure_uri = os.environ.get("COGAME_PLAYER_FAILURE_URI")
            if player_failure_uri:
                await asyncio.to_thread(
                    write_data,
                    player_failure_uri,
                    json.dumps(
                        {
                            "message": summary[:2000],
                            "failed_policy_index": failed[0][0],
                        }
                    ),
                    content_type="application/json",
                    http_method=artifact_method("COGAME_PLAYER_FAILURE_METHOD"),
                )
            session.done = True
            for slot, websocket in list(session.players.items()):
                with suppress(Exception):
                    await websocket.send_json(
                        {
                            **_player_observation(slot),
                            "type": "final",
                            "done": True,
                            "failed": True,
                            "failure_reason": summary,
                        }
                    )
            server.should_exit = True
            await asyncio.sleep(0.2)
            return

        actions = [
            decision.action for decision in decisions if decision.action is not None
        ]
        record = step(session.engine, actions, CONFIG)
        if session.trajectory is not None:
            for slot, decision in enumerate(decisions):
                actual = executed_action(record, slot)
                selected = next(
                    (
                        attempt
                        for attempt in decision.attempts
                        if attempt.attempt_id == decision.selected_attempt_id
                    ),
                    None,
                )
                session.trajectory.record(
                    decision_id=f"round-{record.round}-seat-{slot}",
                    seat=slot,
                    observation=views[slot],
                    prompt=selected.prompt if selected else None,
                    attempts=decision.attempts,
                    executed_action=actual.model_dump(),
                    failure_reason=None,
                    terminal=session.engine.round == CONFIG.rounds,
                )
        session.frames.append({**record.model_dump(), "player_names": PLAYER_NAMES})
        if session.engine.round < CONFIG.rounds:
            await _broadcast_observations()

    results = episode_results(session.engine).model_dump(mode="json")
    if session.trajectory is not None:
        await _drain_trajectory_attempts(loop)
        session.trajectory.finish(
            outcome=results,
            participant_outcomes={
                str(slot): {"score": round(score, 3)}
                for slot, score in enumerate(session.engine.scores)
            },
            completed=session.engine.round == CONFIG.rounds,
        )
        await _write_trajectory()
    logger.info(
        "game finished after %d rounds, scores=%s",
        session.engine.round,
        results["scores"],
    )
    # Artifact writes are blocking HTTP; off the event loop so websocket pings
    # (the hosted certifier probes /global right around game end) still answer.
    await asyncio.to_thread(
        write_data,
        RESULTS_URI,
        json.dumps(results),
        content_type="application/json",
        http_method=artifact_method("COGAME_RESULTS_METHOD"),
    )
    await asyncio.to_thread(
        write_data,
        REPLAY_URI,
        json.dumps(_replay_payload(results)),
        content_type="application/json",
        http_method=artifact_method("COGAME_SAVE_REPLAY_METHOD"),
    )

    session.done = True
    for slot, websocket in list(session.players.items()):
        with suppress(Exception):
            await websocket.send_json(
                {**_player_observation(slot), "type": "final", "done": True}
            )
    # Linger, extending while any global viewer is still attached (bounded).
    linger_until = loop.time() + POST_GAME_LINGER_SECONDS
    hard_stop = loop.time() + POST_GAME_MAX_LINGER_SECONDS
    while loop.time() < hard_stop and (
        loop.time() < linger_until or session.global_viewers > 0
    ):
        await asyncio.sleep(0.5)
    server.should_exit = True
    await asyncio.sleep(0.5)


async def _broadcast_observations() -> None:
    for slot, websocket in list(session.players.items()):
        with suppress(Exception):
            await websocket.send_json(_player_observation(slot))


def _player_observation(slot: int) -> dict[str, Any]:
    view = observation(
        session.engine, CONFIG, slot, PLAYER_NAMES, session.round_seconds
    )
    view["prompt_configuration"] = PROMPT_SELECTION.model_dump()
    if session.engine.round == CONFIG.rounds:
        return {**view, "type": "final", "done": True}
    return view


def _replay_payload(results: dict[str, object] | None) -> dict[str, Any]:
    return {
        "config": {key: value for key, value in RAW_CONFIG.items() if key != "tokens"},
        "player_names": PLAYER_NAMES.copy(),
        "frames": session.frames,
        "results": results,
    }


def _snapshot() -> dict[str, Any]:
    engine = session.engine
    last = engine.history[-1] if engine.history else None
    return {
        "type": "state",
        "round": engine.round,
        "rounds": CONFIG.rounds,
        "stock": round(engine.stock, 2),
        "stock_capacity": CONFIG.stock_capacity,
        "collapse_threshold": CONFIG.collapse_threshold,
        "collapsed": engine.collapsed,
        "collapse_round": engine.collapse_round,
        "scores": [round(score, 2) for score in engine.scores],
        "total_harvested": [round(total, 2) for total in engine.total_harvested],
        "player_names": PLAYER_NAMES.copy(),
        "last_round": last.model_dump() if last else None,
        "connected": sorted(session.players),
        "submitted": sorted(session.pending),
        "started": session.started,
        "paused": session.paused,
        "round_seconds": session.round_seconds,
        "done": session.done,
        "failed": session.failed,
        "failure_reason": session.failure_reason,
    }


if __name__ == "__main__":
    server = uvicorn.Server(uvicorn.Config(app, host=GAME_HOST, port=GAME_PORT))
    server.run()
    if session.failed:
        raise SystemExit(1)
