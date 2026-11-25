"""The same parsed player decision drives hosted, headless, and language training."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from coworld.examples.meadow.game.engine import (
    MeadowConfig,
    RoundAction,
    RoundRecord,
    parse_action,
)
from coworld.examples.meadow.shared.trajectory import Attempt
from coworld.examples.meadow.shared.prompts import PromptSelection


class PlayerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["action"] = "action"
    round: int = Field(ge=0)
    action: RoundAction | None = None
    attempts: list[Attempt] = Field(default_factory=list)
    selected_attempt_id: str | None = None
    failure_reason: str | None = None


def sanitize_error(text: str, *, limit: int = 200) -> str:
    """Collapse whitespace and bound length for private attempt error records."""
    return " ".join(text.split())[:limit]


def parse_json_object(raw: str) -> dict:
    """Parse bare JSON or a single fenced JSON object; raise ValueError on failure."""
    text = raw.strip()
    if not text:
        raise ValueError("empty reply")
    if text.startswith("```"):
        header, sep, rest = text.partition("\n")
        if not sep:
            raise ValueError("unclosed code fence")
        fence_lang = header[3:].strip().lower()
        if fence_lang not in {"", "json"}:
            raise ValueError(
                f"unsupported code fence language: {sanitize_error(fence_lang, limit=32)}"
            )
        if not rest.endswith("```"):
            raise ValueError("unclosed code fence")
        inner = rest[:-3]
        if "```" in inner:
            raise ValueError("multiple code fences")
        text = inner.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("reply is not a single JSON object")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"JSONDecodeError: {sanitize_error(error.msg)}") from error
    if not isinstance(parsed, dict):
        raise ValueError("reply JSON must be an object")
    return parsed


def parse_reply_strict(raw: str, slot: int, config: MeadowConfig) -> RoundAction:
    """Use the shared text parser, then strict engine validation, raising ValueError."""
    return parse_action(parse_json_object(raw), slot, config)


def parse_reply(raw: str, slot: int, config: MeadowConfig) -> RoundAction | None:
    """Use the shipped text parser, then strict engine validation."""
    try:
        return parse_reply_strict(raw, slot, config)
    except ValueError:
        return None


def apply_player_decision(
    decision: PlayerDecision, slot: int, config: MeadowConfig
) -> PlayerDecision:
    """The authenticated socket owns the seat; it does not establish a teacher identity."""
    decision = decision.model_copy(deep=True)
    for attempt in decision.attempts:
        attempt.accepted = False
        if attempt.origin in {"teacher", "human"}:
            attempt.origin = "unknown"
    if len({attempt.attempt_id for attempt in decision.attempts}) != len(
        decision.attempts
    ):
        raise ValueError("duplicate native attempt identity")
    if decision.failure_reason is not None or decision.action is None:
        if decision.selected_attempt_id is not None:
            raise ValueError("failed decision cannot select an attempt")
        decision.action = None
        decision.failure_reason = decision.failure_reason or "missing-action"
        return decision
    try:
        canonical = parse_action(decision.action.model_dump(), slot, config)
    except ValueError:
        decision.action = None
        decision.selected_attempt_id = None
        decision.failure_reason = "invalid-action"
        return decision
    if not decision.attempts:
        attempt = Attempt(
            policy="external-action",
            origin="unknown",
            prompt=None,
            response=canonical.model_dump_json(),
            parsed_action=canonical.model_dump(),
            accepted=True,
            rejection_reason=None,
        )
        decision.attempts = [attempt]
        decision.selected_attempt_id = attempt.attempt_id
    selected = [
        attempt
        for attempt in decision.attempts
        if attempt.attempt_id == decision.selected_attempt_id
    ]
    if len(selected) != 1:
        raise ValueError("submitted decision must select exactly one response")
    attempt = selected[0]
    parsed = parse_reply(
        TypeAdapter(str).validate_python(attempt.response), slot, config
    )
    attempt.parsed_action = parsed.model_dump() if parsed is not None else None
    if attempt.parsed_action != canonical.model_dump():
        attempt.rejection_reason = (
            "model response differs from submitted canonical action"
        )
        decision.action = None
        decision.selected_attempt_id = None
        decision.failure_reason = "player-parser-mismatch"
        return decision
    attempt.accepted = True
    attempt.rejection_reason = None
    decision.action = canonical
    return decision


def decision_prompt(
    view: dict, strategy: str = "", *, selection: PromptSelection = PromptSelection()
) -> list[dict[str, str]]:
    from coworld.examples.meadow.player.policies import system_prompt

    return [
        {
            "role": "system",
            "content": system_prompt(view, strategy, selection=selection),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    key: value
                    for key, value in view.items()
                    if key
                    not in {
                        "type",
                        "round_seconds",
                        "prompt_configuration",
                        "prompt_template",
                    }
                },
                separators=(",", ":"),
            ),
        },
    ]


def executed_action(record: RoundRecord, slot: int) -> RoundAction:
    """Derive the applied command from the settled engine record, including chat and sanctions."""
    return RoundAction(
        harvest=record.demands[slot],
        sanction=next(
            (entry.target for entry in record.sanctions if entry.by == slot), None
        ),
        message=next(
            (entry.text for entry in record.messages if entry.slot == slot), None
        ),
    )


class AttemptProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["attempt"] = "attempt"
    round: int = Field(ge=0)
    attempt: Attempt
