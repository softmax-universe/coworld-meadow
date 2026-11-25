# Meadow Player Protocol

Connect to the game's `/player` WebSocket using the runner-provided `COWORLD_PLAYER_WS_URL` (it already carries your
`slot` and `token` query parameters). The game sends JSON messages; you reply with JSON action envelopes. One action per
round; if you send several before the round settles, the last valid one wins. If you send an invalid action or miss the
round deadline (`round_seconds`), the episode terminates as a policy failure without playing any substitute move or
publishing scores.

## Messages you receive

On connect and after every settled round, one `observation`:

```json
{
  "type": "observation",
  "slot": 2,
  "round": 14,
  "rounds": 60,
  "round_seconds": 20.0,
  "stock": 58.41,
  "stock_capacity": 100.0,
  "regrowth_rate": 0.35,
  "collapse_threshold": 10.0,
  "collapsed": false,
  "max_harvest": 3,
  "num_players": 8,
  "ledger_public": true,
  "sanctions_enabled": true,
  "sanction_cost": 1.0,
  "sanction_burn": 3.0,
  "chat_enabled": true,
  "norm_text": "Posted quota: 1 per player per round.",
  "score": 16.0,
  "your_last_harvest": 1.0,
  "sanctions_received_last_round": 0,
  "last_round_total_harvest": 9.0,
  "messages_last_round": [{ "slot": 5, "text": "hold at 1 each" }],
  "last_round_harvests": [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  "ledger": [
    {
      "slot": 0,
      "name": "P0",
      "total_harvested": 15.0,
      "recent_harvests": [1.0, 1.0, 1.0, 1.0, 1.0],
      "sanctions_given": 0,
      "sanctions_received": 1
    }
  ]
}
```

`last_round_harvests` and `ledger` are present only when `ledger_public` is true (the reputation treatment). The
aggregate `last_round_total_harvest`, your own fields, and signed chat are always visible. Before the first round the
`*_last_*` fields are `null`/empty.

When the episode ends you receive the same shape once with `"type": "final"` and `"done": true`; disconnect after it.

## Actions you send

```json
{
  "type": "action",
  "round": 14,
  "action": { "harvest": 2, "sanction": 4, "message": "stop over-harvesting" }
}
```

- `type` (required): `"action"`.
- `round` (required): integer matching the current observation's `round`. Late replies for earlier rounds are ignored.
- `action` (required for a valid move; `null` with `failure_reason` for an explicit policy failure):
  - `harvest` (required): integer `0..max_harvest`. Non-integer or out-of-range values are rejected.
  - `sanction` (optional): integer player slot to punish (`!= slot`), or `null`. You pay `sanction_cost`, they lose
    `sanction_burn`. Rejected if `sanctions_enabled` is false, out of range, or equal to your own slot.
  - `message` (optional): chat shown to everyone next round, truncated to `chat_max_chars`, or `null`/empty. Non-empty
    messages are rejected when `chat_enabled` is false.

## The game in one paragraph

A shared stock regrows logistically each round (`regrowth_rate * stock * (1 - stock/stock_capacity)`) unless it has
ever dropped below `collapse_threshold` — collapse is permanent. Each round all players harvest simultaneously; if
total demand exceeds the stock, the remainder is split pro-rata. Your score is everything you harvest, minus sanction
costs you pay and burns you receive. The sustainable aggregate near half-capacity is about
`regrowth_rate * stock_capacity / 4` per round — under the default config, roughly one unit per player.
