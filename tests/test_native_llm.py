import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from uuid import uuid4

import pytest

from coworld.examples.meadow.game.engine import MeadowConfig, new_state, observation
from coworld.examples.meadow.headless import PolicyFailureError, run_episode
from coworld.examples.meadow.player.policies import LlmPolicy, SustainablePolicy, system_prompt
from coworld.examples.meadow.shared.decision import apply_player_decision
from coworld.examples.meadow.shared.prompts import PromptSelection
from coworld.examples.meadow.shared.trajectory import Trajectory


def test_native_player_captures_actual_request_response_and_applied_action(monkeypatch):
    requests = []
    call_id = str(uuid4())
    response = {"model": "fixture/served", "content": [{"type": "text", "text": '{"harvest":2}'}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 1},
        "sampling_evidence": {"prompt_token_ids": [1], "completion_token_ids": [2],
            "behavior_log_probs": None, "stop_reason": "eos"}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers["X-Coworld-Player-Slot"], body))
            reply = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("X-Softmax-Llm-Call-Id", call_id)
            self.send_header("X-Coworld-Checkpoint-Sha256", "a" * 64)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}/")
    monkeypatch.setenv("COWORLD_LLM_MODEL", "fixture/requested")
    monkeypatch.setenv("COWORLD_LLM_TEMPERATURE", "0")
    config = MeadowConfig(num_players=2)
    view = observation(new_state(config), config, 1, ["one", "two"], 5)
    view["prompt_configuration"] = PromptSelection().model_dump()
    try:
        decision = LlmPolicy(strategy="PRIVATE OPERATOR SENTINEL").act(view)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(requests) == 1
    path, slot, body = requests[0]
    assert path == "/v1/messages" and slot == "1"
    assert body["model"] == "fixture/requested" and body["temperature"] == 0
    assert body["system"] == system_prompt(view, "PRIVATE OPERATOR SENTINEL")
    attempt = decision.attempts[0]
    assert attempt.request == body and attempt.raw_response == response
    assert str(attempt.platform_call_id) == call_id and attempt.model == "fixture/served"
    assert attempt.model_identity == "a" * 64
    assert attempt.prompt_token_ids == [1] and attempt.sampled_token_ids == [2]
    assert attempt.behavior_logprobs is None
    applied = apply_player_decision(decision, 1, config)
    assert applied.action.harvest == 2 and applied.attempts[0].parsed_action == applied.action.model_dump()


@pytest.mark.parametrize("reply_text", ["not json", '{"harvest": 99}', '{"harvest": true}'])
def test_native_player_invalid_output_fails_without_substitute_action(monkeypatch, reply_text):
    call_id = str(uuid4())
    response = {"model": "fixture/served", "content": [{"type": "text", "text": reply_text}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 1}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            reply = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("X-Softmax-Llm-Call-Id", call_id)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}/")
    monkeypatch.setenv("COWORLD_LLM_MODEL", "fixture/requested")
    monkeypatch.setenv("COWORLD_LLM_TEMPERATURE", "0")
    config = MeadowConfig(num_players=2, rounds=3)
    trajectory = Trajectory(
        episode_id="invalid-native",
        game_version="source-fixture",
        source_revision="a" * 40,
        image_digest=None,
        seed_family="meadow-0",
    )
    try:
        with pytest.raises(PolicyFailureError) as exc_info:
            run_episode(config, [LlmPolicy(), SustainablePolicy()], trajectory=trajectory)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert exc_info.value.round_index == 0 and exc_info.value.slot == 0
    assert exc_info.value.state.round == 0 and exc_info.value.state.history == []
    assert exc_info.value.decisions[0].action is None
    assert exc_info.value.decisions[0].attempts[0].response == reply_text
    assert trajectory.records[-1].status == "failed"
    assert trajectory.records[-1].outcome is None
    assert trajectory.records[-1].participant_outcomes is None


def test_parse_reply_accepts_single_markdown_fence_without_inventing_actions():
    from coworld.examples.meadow.shared.decision import parse_reply

    config = MeadowConfig(num_players=2, chat_enabled=True, sanctions_enabled=True)
    fenced_json = parse_reply('```json\n{"harvest": 2, "message": "hold 1"}\n```', 0, config)
    assert fenced_json is not None
    assert fenced_json.harvest == 2 and fenced_json.message == "hold 1" and fenced_json.sanction is None

    fenced_plain = parse_reply('```\n{"harvest": 1, "sanction": 1}\n```', 0, config)
    assert fenced_plain is not None
    assert fenced_plain.harvest == 1 and fenced_plain.sanction == 1

    assert parse_reply('```json\n{"harvest": 99}\n```', 0, config) is None
    assert parse_reply('```json\n{"message": "no harvest"}\n```', 0, config) is None
    assert parse_reply('Prose before {"harvest": 1}', 0, config) is None
    assert parse_reply('{"harvest": 1} prose after', 0, config) is None
    assert parse_reply('{"harvest": 1}\n{"harvest": 2}', 0, config) is None
    assert parse_reply('```json\n{"harvest": 1}\n```\n```json\n{"harvest": 2}\n```', 0, config) is None
    assert parse_reply('Here is my move:\n```json\n{"harvest": 1}\n```', 0, config) is None


def test_manifest_registers_native_model_player_and_certification_baselines():
    from pathlib import Path

    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "src/coworld/examples/meadow/coworld_manifest_template.json"
    )
    manifest = json.loads(manifest_path.read_text())
    players_by_id = {entry["id"]: entry for entry in manifest["player"]}
    assert "llm" in players_by_id
    llm = players_by_id["llm"]
    assert llm["run"] == ["python", "-m", "coworld.examples.meadow.player.player", "llm"]
    assert llm["env"]["COWORLD_LLM_ENABLED"] == "true"
    assert llm["env"]["COWORLD_LLM_MODEL"] == "anthropic/claude-haiku-4.5"
    assert llm["env"]["COWORLD_LLM_TEMPERATURE"] == "1"
    assert "COWORLD_LLM_ENDPOINT" not in llm["env"]
    assert manifest["certification"]["game_config"]["round_seconds"] == 20
    assert manifest["game"]["runnable"]["source_url"].startswith(
        "https://github.com/softmax-universe/coworld-meadow/"
    )
    assert llm["source_url"].startswith("https://github.com/softmax-universe/coworld-meadow/")

    cert_ids = [slot["player_id"] for slot in manifest["certification"]["players"]]
    assert "llm" in cert_ids
    for baseline_id in ("sustainable", "greedy", "reciprocator", "deterrable", "enforcer", "random"):
        assert baseline_id in players_by_id
        assert baseline_id in cert_ids
        assert "COWORLD_LLM_ENABLED" not in players_by_id[baseline_id].get("env", {})


def test_system_prompt_teaches_minimal_action_and_other_player_sanction():
    cfg = MeadowConfig(num_players=4, sanctions_enabled=True, chat_enabled=True)
    obs = observation(new_state(cfg), cfg, 0, ["P0", "P1", "P2", "P3"], 20)
    prompt = system_prompt(obs)
    assert '{"harvest": <int 0..3>}' in prompt
    assert "omit when unused" in prompt
    assert "one other player per round (never your own slot)" in prompt
    assert "or null" not in prompt
    assert "may be empty" not in prompt

    minimal_cfg = MeadowConfig(num_players=4, sanctions_enabled=False, chat_enabled=False)
    minimal_obs = observation(new_state(minimal_cfg), minimal_cfg, 0, ["P0", "P1", "P2", "P3"], 20)
    minimal_prompt = system_prompt(minimal_obs)
    assert '{"harvest": <int 0..3>}' in minimal_prompt
    assert "Optional key" not in minimal_prompt
    assert "sanction" not in minimal_prompt
    assert "message" not in minimal_prompt


@pytest.mark.parametrize(
    "status,body,expected_prefix",
    [
        (503, '{"error": "service unavailable"}', "HTTP 503"),
        (200, "not valid json", "JSONDecodeError:"),
        (200, '{"unexpected": true}', "ValidationError:"),
    ],
)
def test_native_http_503_and_malformed_response_produce_no_score(
    monkeypatch, status, body, expected_prefix
):
    call_id = str(uuid4())

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("X-Softmax-Llm-Call-Id", call_id)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}/")
    monkeypatch.setenv("COWORLD_LLM_MODEL", "fixture/requested")
    monkeypatch.setenv("COWORLD_LLM_TEMPERATURE", "1")
    config = MeadowConfig(num_players=2, rounds=3)
    trajectory = Trajectory(
        episode_id="http-or-malformed-native",
        game_version="source-fixture",
        source_revision="a" * 40,
        image_digest=None,
        seed_family="meadow-0",
    )
    try:
        with pytest.raises(PolicyFailureError) as exc_info:
            run_episode(config, [LlmPolicy(), SustainablePolicy()], trajectory=trajectory)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    decision = exc_info.value.decisions[0]
    assert decision.action is None
    assert decision.failure_reason == "native-call-failure"
    assert decision.attempts[0].rejection_reason.startswith(expected_prefix)
    assert trajectory.records[-1].status == "failed"
    assert trajectory.records[-1].outcome is None
    assert trajectory.records[-1].participant_outcomes is None


def test_self_sanction_records_exact_engine_error_in_private_attempt(monkeypatch):
    call_id = str(uuid4())
    response = {
        "model": "fixture/served",
        "content": [{"type": "text", "text": '{"harvest": 1, "sanction": 0}'}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 5},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("X-Softmax-Llm-Call-Id", call_id)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}/")
    monkeypatch.setenv("COWORLD_LLM_MODEL", "fixture/requested")
    config = MeadowConfig(num_players=2, rounds=3, sanctions_enabled=True)
    view = observation(new_state(config), config, 0, ["one", "two"], 20)
    view["prompt_configuration"] = PromptSelection().model_dump()
    try:
        decision = LlmPolicy().act(view)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert decision.action is None
    assert decision.failure_reason == "invalid-native-reply"
    assert decision.attempts[0].rejection_reason == "invalid sanction target: 0"
