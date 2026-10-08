# Tic-Tac-Toe with Q-Learning

A Reinforcement Learning Lab CA mini project: train a tabular Q-learning agent and play Tic-Tac-Toe in a **Python notebook**. The board sits on the left; the Q-values for each AI decision appear on the right.

The agent learns from game outcomes and selects moves from its Q-table. Minimax is used during training, never to choose moves in the playable notebook.

## Explain the project to faculty

**One-minute explanation:** “My project trains a Tic-Tac-Toe agent with tabular Q-learning. A state is the nine-square board written from the agent's point of view. An action is one empty square. The Q-table stores an estimated future reward for each state–action pair. During 160,000 self-contained training games, the agent explores legal moves, observes the opponent's reply, receives a reward for a win, draw, or loss, and updates one Q-value using the Bellman equation. After training, it stops exploring and chooses the legal move with the highest Q-value. I compare it with random, tactical, and minimax opponents, and I show the learned values and decisions in an offline Python notebook.”

The complete flow is:

```text
Board state → legal empty squares → ε-greedy AI action → opponent reply
           → reward and next AI state → Q-value update → repeat until game ends
```

### Q-table basics

Think of a Q-table as a lookup table with **one row per board state** and **one column per possible square**. `Q(s, a)` estimates how useful action `a` is when the agent sees state `s`. A high value is preferred, but **a Q-value is expected discounted reward, not a win probability**. The table is learned from repeated play; it is not filled with hand-written Tic-Tac-Toe rules.

The board uses row-major indices:

```text
Index:     0 | 1 | 2       Displayed plot labels: 1 | 2 | 3
           3 | 4 | 5                              4 | 5 | 6
           6 | 7 | 8                              7 | 8 | 9
```

The state string uses `0 = empty`, `1 = AI`, and `2 = opponent`. For example, if the human plays X in the top-left and the AI is O, the AI sees `200000000`. Its row has nine possible columns, but square `0` is occupied, so only squares `1` through `8` are considered. In a final trained run, the center (`action 4`, or square 5 on the plot) is the best reply with a Q-value around `+0.404` for this opening. Values can vary during training; read the current notebook output for the exact value.

| Item | Meaning |
|---|---|
| `Q(s, a)` | Estimated discounted return if the AI takes legal action `a` in state `s` and then follows its learned policy. |
| Unseen state/action | Starts at `0.0`; its estimate changes when training visits it. |
| Occupied square | Illegal action; shown as occupied in the UI and excluded from selection and the Bellman maximum. |
| Highest legal Q-value | Greedy action used after training; ties are broken randomly. |
| Canonical state | Rotated/reflected boards share one stored Q-table row, reducing duplicate learning. The displayed values are mapped back to the original board coordinates. |

The Q-learning update in [`QLearningAgent.update`](tictactoe/agent.py) is:

```text
target = reward + γ × max Q(next_state, legal_next_action)  # nonterminal
target = reward                                            # terminal
Q(state, action) ← Q(state, action) + α × (target − Q(state, action))
```

Here `α = 0.15` controls how far the old value moves toward the target, and `γ = 0.97` controls how much future reward matters. For a simple terminal example, if an action's old Q-value is `0.20` and it wins immediately, the target is `+1`; the new value is `0.20 + 0.15 × (1 − 0.20) = 0.32`. For a nonterminal move, the agent also considers the best legal Q-value at its **next turn after the opponent replies**. This makes the state transition match the turn-by-turn game.

### Code walkthrough from start to finish

1. [`tictactoe/core.py`](tictactoe/core.py) defines the board as a nine-element tuple, checks wins and draws, rejects illegal moves, encodes a state from the AI's point of view, and provides random, tactical, and minimax training opponents. `canonicalize` maps the eight rotations/reflections of a board to one shared table key.
2. [`tictactoe/agent.py`](tictactoe/agent.py) owns `self.q`, a dictionary from canonical state strings to nine Q-values. `values` maps stored values back to visible square positions. `select_action` uses ε-greedy selection during training and ignores occupied squares. `update` applies the equation above to the action just taken.
3. The [one-cell notebook](notebooks/TicTacToe_All_In_One.ipynb) trains for `EPISODES = 160_000`, alternating whether the AI plays X or O. For each AI move it encodes the current board, selects a move, lets the opponent reply if the game continues, calculates the reward, and updates Q. Training opponents are chosen per opponent turn: 25% random, 15% tactical, and 60% minimax.
4. Exploration starts at `ε = 1.0`, falls linearly to `0.03` over 85% of training, and stays at `0.03`. This is one **global schedule** for all starting squares. It lets the AI try unfamiliar moves early and favor learned moves later.
5. Every 1,000 training games, the notebook records reward, wins/draws/losses, steps, TD error, and Q-tables for **all nine human openings**. Every 10,000 games, it separately tests 100 greedy games for each opening against random human follow-up moves. These evaluation games do **not** update the Q-table.
6. After training, the notebook tests the greedy agent against random, tactical, and minimax opponents and runs 1,000 games for each human opening. It displays learning curves, opening comparisons, and action-value graphs.
7. In the playable [`NotebookTicTacToe` class](scripts/build_one_cell_notebook.py), the human clicks a square; the trained agent reads that board's Q-values, chooses a legal maximal-Q square with `ε = 0`, and updates the board. The 3×3 value panel and bar chart explain that exact decision. The widget source is generated into the one-cell notebook by [`scripts/build_one_cell_notebook.py`](scripts/build_one_cell_notebook.py).

The reward is `+1` for an AI win, `+0.3` for a draw, `−1` for an AI loss, and `0` before the game ends. A draw has a positive reward because drawing a perfect opponent is preferable to losing. The opponent policy is **part of training/evaluation**, while the deployed agent's move comes from its learned Q-table.

### How to explain the graphs

| Graph | What to tell faculty |
|---|---|
| Reward vs episode | Mean terminal reward in each 1,000-game training window; rising values show better outcomes as exploration falls. |
| Success rate vs episode | Fraction of training games with a win or draw. These games still include exploration, so this is different from final greedy evaluation. |
| Steps per episode | Board moves by both players and decisions by the AI. More moves can mean the AI survives to a draw rather than losing early. |
| TD error vs episode | Mean absolute gap between the Q-learning target and the previous Q-value. Smaller updates suggest the estimates are settling; they do not prove convergence. |
| Epsilon vs episode | The predetermined exploration schedule. It is identical for every starting square and is **not** a position-performance plot. |
| Opening-specific win-rate curves | Nine separate greedy evaluations over training. These show the actual effect of the human's first square on AI performance. |
| Opening Q-value curves and action bars | Show how the best estimated reply changes and which legal square the AI chooses for each opening. Green is the chosen action. |

**Likely viva questions:** “Why Q-learning?” Tic-Tac-Toe has a small, discrete state/action space, making the table easy to inspect. “Why ε-greedy?” Exploration prevents the agent from only repeating its first promising move. “Why does minimax appear?” It is a strong training opponent and benchmark, not the algorithm used by the playable AI. “Does a higher Q mean a higher win percentage?” No; it estimates discounted reward under the training process. “Did it converge?” The curves stabilize empirically in this finite run, but a constant learning rate and exploration floor do not justify a mathematical convergence claim.

## Play in Python

Open [TicTacToe_All_In_One.ipynb](notebooks/TicTacToe_All_In_One.ipynb) in Jupyter and run its **one Python code cell**. It trains the agent and records a 3×3 Q-table **for each of the nine human opening squares every 1,000 games**. The checkpoint tables are grouped by opening in tabs. After training, the plots appear before the game: reward, success rate, and steps versus episode; outcomes and best Q-value by human opening; the best Q-value's training progress and win-rate progress for all nine openings; and nine AI-decision bar charts, one per opening. Expand the panels for the full training table and Q-table snapshots. The notebook also evaluates against random, tactical, and minimax opponents, then runs **1,000 games for each of the nine human first squares** against random X replies. These 9,000 games evaluate the trained, greedy policy; they do not update it. The playable 3×3 board sits beside its decision Q-values and an updating bar chart of all nine actions. Green marks the chosen action, blue marks other legal actions, and gray marks occupied squares. Choose X or O and use **New game** to reset. The notebook uses Python and `ipywidgets` for the game; a small inline style keeps button text black in dark notebook themes.

For a class presentation, open the [complete commented Python code](notebooks/TicTacToe_All_In_One_Commented.py). It contains the same single-cell program as the notebook, with brief comments for the rules, Q-table, training, evaluation, plots, and game. Paste the file into a Jupyter code cell to run the interactive UI; the `.ipynb` version is ready to run and includes saved plots.

Open the plots directly: [training curves](artifacts/plots/notebook_training_plots.png), [results by human opening](artifacts/plots/notebook_opening_graphs.png), [Q-value progress for all openings](artifacts/plots/notebook_opening_q_progress.png), and [AI decisions for all openings](artifacts/plots/notebook_ai_decisions_all_openings.png). The decision chart in the running notebook updates after every AI move.

The notebook also displays five separate standard RL graphs before the existing diagrams: **reward vs episode**, **success rate vs episode**, **steps per episode**, **mean absolute TD error vs episode**, and **exploration rate (epsilon) vs episode**. Each uses non-overlapping 1,000-game training windows. Reward and success are exploratory training outcomes; TD error measures the size of Q-learning updates, not prediction accuracy.

Epsilon is a **global training setting**, so its schedule is the same for every starting square. To show how the starting square affects learning, the notebook separately evaluates the greedy AI every 10,000 training games: 100 games for each of the nine human openings against random follow-up moves. The resulting [opening-specific win-rate curves](artifacts/plots/notebook_opening_win_rate_progress.png) plot performance by starting square. The existing Q-value progress and AI-decision diagrams remain included.

Direct PNGs: [reward](artifacts/plots/notebook_reward_vs_episode.png), [success rate](artifacts/plots/notebook_success_vs_episode.png), [steps](artifacts/plots/notebook_steps_vs_episode.png), [TD error](artifacts/plots/notebook_td_error_vs_episode.png), and [epsilon](artifacts/plots/notebook_epsilon_vs_episode.png).

```bash
python -m pip install -r requirements.txt
jupyter notebook notebooks/TicTacToe_All_In_One.ipynb
```

The default 160,000-game run uses the same training configuration as the project and was checked against all 627 saved canonical states. Edit `EPISODES` in the notebook for a shorter demonstration. The interface uses `ipywidgets` and runs locally with Python and Jupyter.

![Offline game with learned action values](artifacts/screenshots/offline_game_qvalues.jpg)

## Optional browser version

1. [Download the project ZIP](https://github.com/shreejaykurhade/tic-tac-toe-qlearning-lab/archive/refs/heads/main.zip) and extract it.
2. Double-click **`TicTacToe_Offline.html`** to open it in Chrome, Edge, Firefox, or another modern browser.
3. Choose X or O and play. X always starts.

The standalone HTML includes the UI and trained model. **No internet, Python installation, server, or API key is needed to play.** Keep just this file if you only want the game.

Features include a responsive layout, keyboard navigation, a per-opponent scorecard, a random baseline opponent, and a toggle showing the agent's learned action values. The model remains fixed while you play. If browser storage is restricted, gameplay still works, but scores may not persist.

## Training results

The included model was trained for **160,000 episodes**, alternating equally between X and O, with seed **42**. Final evaluation used a separate seed, zero exploration, and **2,000 games per role per opponent**: 12,000 games total.

| Opponent | Games | Wins | Draws | Losses | Win rate | Non-loss rate |
|---|---:|---:|---:|---:|---:|---:|
| Random | 4,000 | 3,769 | 231 | 0 | 94.23% | 100% |
| Tactical | 4,000 | 676 | 3,324 | 0 | 16.90% | 100% |
| Minimax | 4,000 | 0 | 4,000 | 0 | 0% | 100% |

The zero-table baseline won 44.73% against random play and lost 87.45% against minimax. A draw is the best guaranteed outcome against perfect Tic-Tac-Toe play, so non-loss rate matters alongside win rate.

An exact adversarial game-tree audit found **zero worst-case loss probability from the empty board for both roles**, including all maximal-Q tie choices. This conclusion applies to the included saved policy and its greedy tie rule; it does not describe arbitrary forced board positions or exploratory play.

The model contains **627 canonical states**, expanded to **4,520 board orientations** for the browser. Detailed results, role breakdowns, seeds, and run provenance are in [evaluation.json](artifacts/evaluation.json) and [training_summary.json](artifacts/training_summary.json).

## How the learner works

- **State:** nine cells, encoded relative to the agent: `0` empty, `1` self, `2` opponent.
- **Action:** a legal empty square, indexed 0–8 in row-major order.
- **Reward:** win `+1`, draw `+0.3`, loss `−1`, ongoing transition `0`.
- **Transition:** the agent move plus the opponent response, so the next nonterminal state is the same agent's next decision.
- **Learning:** epsilon-greedy tabular Q-learning with legal-action masking and shared rotations/reflections.
- **Deployment:** select a legal maximal-Q action, breaking ties uniformly within `1e-12`. Unknown positions use a clearly labeled random fallback.

```text
Q(s,a) ← Q(s,a) + α [r + γ max_legal Q(s′,a′) − Q(s,a)]
```

Terminal transitions use `r` alone, without bootstrapping.

| Parameter | Value |
|---|---|
| Learning rate α | 0.15 |
| Discount factor γ | 0.97 |
| Exploration ε | Linear 1.00 → 0.03 over the first 85% of episodes; then 0.03 |
| Training episodes | 160,000 |
| Training opponent mixture | 25% random, 15% tactical, 60% minimax |
| Training seed | 42 |
| Log / checkpoint intervals | 1,000 / 5,000 episodes |

The opponent policy is sampled independently **at each opponent move**. Minimax is used as a training opponent and evaluation benchmark, not as the deployed agent. The tactical opponent takes an immediate win, blocks an immediate threat, then prefers the center, corners, and remaining edges.

## Run the Python experiment

Use **Python 3.10 or newer**. A CPU is sufficient.

```bash
git clone https://github.com/shreejaykurhade/tic-tac-toe-qlearning-lab.git
cd tic-tac-toe-qlearning-lab
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m tictactoe.train --episodes 160000 --seed 42 --output artifacts --evaluation-games 2000
python scripts/build_offline.py
```

The final command rebuilds the standalone HTML and `artifacts/model.js` using your newly trained model. Training alone does not replace the model embedded in the standalone game.

Training writes the Q-table, summary, final evaluation, CSV logs, and four PNG plots into `artifacts/`. The game rules and learning engine use the Python standard library; Matplotlib and NumPy support plotting.

For a short demonstration, reduce `--episodes`. A shorter run can produce different performance from the included model. Repeat multiple seeds when comparing algorithms; the reported results come from one training seed.

## Notebook

The [one-cell Python notebook](notebooks/TicTacToe_All_In_One.ipynb) is the simplest way to train, examine the learning results, and play. It includes saved training tables and plots for the lab writeup. Its game Q-value panel updates with the agent's latest decision.

The earlier run's full Q-table history remains available as [1,600 snapshots every 100 games](artifacts/q_table_every_100.jsonl.gz) and a [checkpoint CSV](artifacts/q_table_index_every_100.csv). These are separate research artifacts. The playable notebook includes opening-state snapshots every 1,000 games and shows the Q-values relevant to each live decision.

[Open the one-cell notebook in Colab](https://colab.research.google.com/github/shreejaykurhade/tic-tac-toe-qlearning-lab/blob/main/notebooks/TicTacToe_All_In_One.ipynb)

The older notebook below offers the same project as a guided sequence of smaller cells with additional graphs and a browser demonstration.

Open [TicTacToe_Q_Learning.ipynb](notebooks/TicTacToe_Q_Learning.ipynb) in Jupyter and run all cells in order. It embeds the project source, creates a temporary experiment folder, trains the agent, displays metrics and plots, runs the Python tests, and embeds the complete game in an iframe.

The included notebook was **executed locally: eight code cells completed with zero errors**. It is also compatible with Google Colab; a cloud execution is not claimed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shreejaykurhade/tic-tac-toe-qlearning-lab/blob/main/notebooks/TicTacToe_Q_Learning.ipynb)

Jupyter requires its packages to be installed locally. Once those dependencies are present, the notebook uses embedded source and does not need a project download at execution time. Running the notebook creates a new model in its temporary folder and does not overwrite the repository artifacts.

## Learning curves

![Reward vs episode](artifacts/plots/reward_vs_episode.png)

![Non-loss rate vs episode](artifacts/plots/success_vs_episode.png)

![Steps per episode](artifacts/plots/steps_vs_episode.png)

Reward is shown in 1,000-episode exploratory training windows. Checkpoint non-loss rates use the greedy policy, 200 games per role per opponent. Longer episodes can indicate successful defense and full-board draws, rather than worse performance. Stable finite-run results with constant α are not a general proof of Q-learning convergence.

## College report

The report follows the Lab CA structure: problem definition, RL formulation, implementation, parameters, demonstration, results, graphs, analysis, limitations, future scope, conclusion, references, viva questions, and code appendix.

- [Editable Word report](report/LAB_CA_Report.docx)
- [PDF report](report/LAB_CA_Report.pdf)
- [Markdown report](report/LAB_CA_Report.md)

Fill in your student name, roll number, division/batch, and performance date before submission. Review the content against your college's formatting requirements.

The report builder reads the measured artifacts. To regenerate it, install `python-docx`, `reportlab`, and `Pillow`, then run `python scripts/build_report.py`. Those are optional report-authoring dependencies, not game dependencies.

## Project structure

```text
TicTacToe_Offline.html           Portable game with embedded trained model
index.html / styles.css / app.js Editable browser UI source
tictactoe/
  core.py                       Rules, symmetry transforms, opponent policies
  agent.py                      Q-learning and model export/load
  train.py                      Training, experiment output, plots
  evaluate.py                   Baseline evaluation and adversarial audit
notebooks/                      Executed, self-contained notebook
artifacts/                      Model, Q-table snapshots, results, plots, screenshots
report/                         Lab report in Word, PDF, and Markdown
scripts/                        Offline, notebook, and report builders
tests/                          Python and JavaScript checks
```

## Verification

```bash
python -m unittest discover -s tests -v
node --test tests/frontend.test.cjs
```

The Python checks cover all winning lines, draws, legal actions, terminal/nonterminal updates, symmetry/export agreement, random ties, and reproducibility. The JavaScript checks cover legal move selection, repeated clicks, restart races, playing as O, terminal scoring, fallback behavior, keyboard navigation, and restricted local storage. Node.js is needed only for these frontend checks.

The interface was also reviewed in a local browser preview at desktop and mobile sizes. The standalone file contains no remote game assets. Browser automation could not open `file://` directly, so the visual check served the same bundled HTML from localhost; the game itself requires no server.

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. [MIT Press](https://mitpress.mit.edu/9780262039246/reinforcement-learning/).
2. Watkins, C. J. C. H., & Dayan, P. (1992). “Q-Learning.” *Machine Learning*, 8, 279–292. [Paper](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).
3. [Python documentation](https://docs.python.org/3/) and [Matplotlib documentation](https://matplotlib.org/stable/).
