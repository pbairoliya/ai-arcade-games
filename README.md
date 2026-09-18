# ai-arcade-games

A playable Snake implementation in PyGame, plus the scaffolding for a Deep Q-Network agent
that learns to play it. Written **July 2023**; published to GitHub later, so the commit dates
trail the work.

**Status: unfinished.** The reinforcement-learning plumbing is written and the training loop
runs end to end, but the Q-network itself was never implemented — `Agent.model` and
`Agent.trainer` are still `None`. Preserved as-is rather than quietly polished.

## What's here

| File | Status | Role |
|---|---|---|
| `game.py` | complete | Snake as an RL environment — `play_step(action)` returns `(reward, done, score)` |
| `snake_game.py` | complete | The human-playable version of the same game |
| `agent.py` | scaffold | State encoding, replay memory, epsilon-greedy action selection, training loop |

## The part I'd keep

The state encoding in `Agent.get_state()` is the design decision worth reading. The agent
never sees the board — it sees an 11-value boolean vector: danger straight / right / left,
the four direction flags, and the food's position relative to the head. Feeding a network
that instead of raw pixels is what makes this trainable on a laptop rather than a GPU cluster.

Rewards are sparse by design: `+10` for food, `-10` for dying, `0` otherwise.

## To finish it

Implement `Linear_QNet` (2 layers is enough for an 11-value state) and a `QTrainer` exposing
`train_step()`, then assign both in `Agent.__init__`. Everything calling them is already written.

## Running

```bash
pip install pygame numpy torch
python snake_game.py   # play it yourself — works today
python agent.py        # training loop — needs the model implemented first
```
