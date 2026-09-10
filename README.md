# Ms. Pac-Man DQN — Class 3

Training a Deep Q-Network to play Ms. Pac-Man, using the supplied
[`pacman-dqn`](https://github.com/pepealonso95/pacman-dqn) notebook. This repository holds the
executed notebook, the evidence from the final run, and an account of what the agent actually
learned — including what it did not.

<!-- RESULTS_HEADLINE -->

## Contents

| File | What it is |
|---|---|
| [`pacman_dqn.ipynb`](pacman_dqn.ipynb) | The executed notebook, outputs intact |
| [`results/config.json`](results/config.json) | Every setting and package version used |
| [`results/comparison.json`](results/comparison.json) | All ten evaluation scores, before and after |
| [`results/training.csv`](results/training.csv) | Per-episode score, loss, decisions, updates, timing |
| [`results/training_summary.json`](results/training_summary.json) | Totals for the run |

## How to reproduce

1. Open [`pacman_dqn.ipynb`](pacman_dqn.ipynb) in Google Colab, or locally on a Python 3.11–3.13 kernel.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all.** The notebook installs its own packages and detects CUDA, MPS, or CPU.

The three settings live in the Section 1 cell. Nothing else needs editing to reproduce this run.

## What the agent observes, does, and is rewarded for

**Observations — four game screens.** The agent never sees the game the way a person does. Each
frame is cropped and shrunk to an 84×84 grayscale image, and the four most recent frames are
stacked into one input. Four frames rather than one because a single still picture cannot show
motion: from one frame you cannot tell whether a ghost is closing in or moving away. The stack is
what makes direction and speed visible.

**Actions — joystick moves.** The agent picks one of the Atari joystick positions: no-op, the four
directions, and the four diagonals. It chooses once every four emulator frames, so roughly fifteen
decisions per second of game time. There is no notion of "eat the pellet" or "run from the ghost"
anywhere in the code — only joystick positions.

**Reward — game points.** Every point Ms. Pac-Man scores is the reward signal. Nothing else is
supplied: no bonus for surviving, no penalty for dying, no hand-written advice about ghosts. The
agent has to infer that dying is bad purely from the points it stops collecting afterwards.

One wrinkle that matters, and it drives the limitation below: during learning the reward is
clipped to the range −1 to +1 by `float(np.clip(reward, -1, 1))`. Scores shown in this README are
real game points, but the learner only ever saw +1 or 0.

## The three settings

| Setting | Default | Mine |
|---|---|---|
| Exploration | 0.20 | **0.10** |
| Episodes | 100 | **400** |
| Learning rate | 0.0001 | **0.0001** (kept) |

**Exploration = 0.10.** The default of 0.20 is too high for this environment, and the reason is in
the environment constructor: `repeat_action_probability=0.25`. Sticky actions are on, so the
emulator ignores the agent's chosen action and repeats the previous one a quarter of the time.
Stacking 20% epsilon-greedy noise on top of that means the agent's intended move survives only
about 60% of steps — not enough control to execute the kind of sustained route through a maze
corridor that scoring requires. Exploration here is also *constant*: the code reads
`epsilon = 1.0 if total_steps < WARMUP_STEPS else EXPLORATION`, with no decay, so whatever I pick
is a permanent noise floor rather than a starting point. I chose 0.10 because evaluation runs at
0.05, and keeping the training state distribution close to the evaluation distribution matters
more here than extra coverage. Coverage is already protected by the 1,000 fully-random warm-up
decisions, which my enlarged replay buffer retains for the whole run.

**Episodes = 400.** Four times the notebook's default. At 3,000 decisions per game this budgets
roughly 560,000 decisions, about 2.2 million emulator frames, and around 140,000 gradient updates.
Episodes are the most direct lever on final score, and 400 was the most I could fit inside a
free-tier Colab session without hitting the idle-disconnect limit. This is still tiny by Atari
standards — published DQN results use 50 million frames, more than twenty times this budget — so
the honest expectation is a partially-trained agent, not a good one.

**Learning rate = 0.0001, kept at the default.** A deliberate choice, not an omission. Adam at
1e-4 is the well-tested Atari value, and two properties of this notebook argue against raising it.
First, the network is plain DQN with no Double-DQN correction, so Q-value overestimation compounds
unchecked. Second, and decisively, evaluation reloads `trained.pt` — the **final** network, not the
best checkpoint. A late-training divergence is therefore unrecoverable and would cost the entire
result. Trading a modest speedup for that risk was not worth it. Holding the learning rate fixed
also isolates the effect of the two settings I did change.

## The one other change I made: replay capacity

The assignment permits tuning other hyperparameters with an explanation. I changed exactly one:

```python
REPLAY_CAPACITY = 50000   # was 5000
```

The default of 5,000 transitions is the notebook's most serious limitation. At roughly 35 KB per
transition — the notebook computes this itself as `REPLAY_CAPACITY * 5 * 84 * 84` — the buffer
holds under two episodes of experience. Minibatches drawn from it are therefore almost on-policy
and heavily correlated with one another, which defeats the purpose of experience replay: breaking
that correlation is why DQN has a replay buffer at all. With a two-episode buffer the network
continually overwrites what it learned from earlier states, the textbook catastrophic-forgetting
setup.

At 50,000 the buffer holds roughly seventeen episodes and costs 1,682 MiB against Colab's ~12.7 GB.
It is the cheapest large improvement available in this notebook.

No evaluation setting was touched. `EVAL_SEEDS` and `EVAL_EXPLORATION` remain
`[101, 202, 303, 404, 505]` and `0.05`, identical before and after training, as required.

I also added one optional cell that mounts Google Drive and changes the working directory, so that
checkpoints written every 25 episodes would survive a Colab disconnect. It affects no
hyperparameter and no evaluation setting — only where files are written.

## What I expected, and what I observed

**Written before training started, and left unedited afterwards.**

- **Baseline (untrained).** I expected roughly **150–350**. Worth being precise about what the
  baseline is: it is an untrained *network*, not a random-action agent. A randomly initialized CNN
  tends to commit to one output, so with sticky actions and 5% exploration it often behaves like an
  agent holding the joystick in a single direction. I expected it to score at or below a genuinely
  random policy, not above it.
- **Trained.** I expected roughly **600–1,200** mean. Ms. Pac-Man gives dense rewards, so
  pellet-eating is learnable inside this budget; reliable ghost avoidance is not, and the
  power-pellet strategy is unreachable for the reason given under *Limitation*.
- **Training loss.** I expected it to be noisy and non-monotonic, and to correlate poorly with
  score. Falling loss would not be evidence of better play.
- **The dominant risk.** Because evaluation scores the final network rather than the best
  checkpoint, I expected the possibility of finishing below the mid-run peak.

<!-- OBSERVED -->

<!-- RUN_FACTS -->

<!-- EVALUATION_TABLE -->

<!-- GAMEPLAY -->

<!-- TRAINING_PLOT -->

## One limitation

**Reward clipping makes the agent blind to what is actually worth points.** During training every
reward passes through `float(np.clip(reward, -1, 1))`. A 10-point pellet becomes +1. A ghost eaten
after a power pellet is worth 200, then 400, 800, and 1,600 points — and also becomes +1. Four
ghosts eaten in one power-pellet window are worth 3,000 points, more than an entire round of
diligent pellet-eating, and the learner cannot distinguish that from swallowing a single dot.

The consequence is structural rather than incidental: the agent has no gradient whatsoever
pointing toward the power-pellet-then-hunt strategy that separates high Ms. Pac-Man scores from
mediocre ones. It can only learn "move toward things that give reward, and avoid dying." Every
score ceiling in this experiment follows from that. Clipping exists for a good reason — it
stabilizes gradients across games with wildly different point scales — but on Ms. Pac-Man
specifically it discards the game's central strategic decision.

This is also why a falling training loss says very little here. The loss measures how well the
network predicts its own clipped targets, not whether the agent plays well.

## The next experiment I would run

**Change one setting: replace reward clipping with reward scaling.** Instead of
`np.clip(reward, -1, 1)`, divide the raw reward by a constant — dividing by 100 maps a pellet to
0.1 and a fourth ghost to 16.0. That preserves the gradient stability clipping was introduced for,
while restoring the *relative* value of ghost-eating that clipping destroys.

I would change this one thing and hold everything else fixed, because it targets the specific
limitation above rather than spending more compute on the same blind objective. The prediction
that makes it falsifiable: mean evaluation score rises, and the trained GIF shows the agent moving
toward power pellets rather than treating them as ordinary food. The risk worth naming is that
larger reward magnitudes destabilize the Q-targets — exactly the failure clipping was introduced
to prevent — so the run could come out worse, and that result would be worth reporting too.

## Where the checkpoints are

Model checkpoints (`.pt`) are not in this repository; they are large and add nothing a grader
needs. They are kept in the full run ZIP saved locally, alongside every file reproduced here. The
evidence in `results/` is copied from that ZIP unmodified.
