# ai-arcade-games

Arcade games that learn to play themselves. Currently: **Snake**, played by a Deep Q-Network
agent that starts knowing nothing and trains from its own collisions.

*(Written July 2023; published to GitHub later, so the commit dates trail the work.)*

## How it works

Three pieces, deliberately separated so the learning is swappable from the game:

| File | Role |
|---|---|
| `snake_game.py` | The environment — board state, movement, collision, food placement, reward |
| `agent.py` | The learner — state encoding, replay memory, epsilon-greedy exploration, training step |
| `game.py` | A human-playable build of the same game, for comparison |

The agent sees an 11-value state vector (immediate danger straight/right/left, current
direction, and the food's relative position) rather than raw pixels, which is what makes
training feasible on a laptop. Reward is `+10` for food, `-10` for dying, `0` otherwise.
Exploration decays with games played, so early games are nearly random and later ones are
nearly greedy.

## Running

```bash
pip install torch pygame numpy
python agent.py      # train the agent
python game.py       # play it yourself
```

## What I took from it

The interesting failure was reward shaping: with a distance-to-food reward the snake learned
to circle the food forever rather than eat it, because orbiting scored better than the risk of
approaching. Sparse rewards trained slower and worked.
