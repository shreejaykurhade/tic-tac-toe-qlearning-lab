"""Create a self-contained, offline-capable notebook with the exact project code."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    files = list((ROOT / 'tictactoe').glob('*.py')) + list((ROOT / 'tests').glob('*.py'))
    files += [ROOT / name for name in ('index.html', 'styles.css', 'app.js', 'scripts/build_offline.py', 'assets/favicon.svg')]
    source = {str(path.relative_to(ROOT)).replace('\\', '/'): path.read_text(encoding='utf-8') for path in files}
    setup = '''# All project source is embedded below. No download, API, or internet connection is used.
import os, sys, json, tempfile
from pathlib import Path
from IPython.display import display, HTML, Image, Markdown
PROJECT_FILES = ''' + repr(source) + '''
WORK = Path(tempfile.mkdtemp(prefix="tic_tac_toe_qlearning_"))
for relative, text in PROJECT_FILES.items():
    target = WORK / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
os.chdir(WORK)
sys.path.insert(0, str(WORK))
print("Local experiment folder:", WORK)
print("Python:", sys.version.split()[0])
'''
    cells = [
        nbf.v4.new_markdown_cell('''# Tic Tac Toe agent using tabular Q-learning
## Reinforcement Learning Lab CA mini project
Train a real agent, evaluate it as both X and O, plot the results, and play its learned policy.

This notebook includes all project source. It runs locally in Jupyter with Python, NumPy and Matplotlib, and is compatible with Google Colab. A CPU is sufficient. **Run all cells in order.** Nothing is downloaded. The browser game uses the saved table and does not train during human play.

Student name: __________  |  Roll number: __________  |  Division/batch: __________  |  Performance date: __________
'''),
        nbf.v4.new_markdown_cell('### 1. Prepare the included project\nThe setup cell restores the Python package, tests and browser UI into a temporary local folder. The readable source files are also provided in the project folder.'),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell('''### 2. Configure the experiment
An agent decision covers its own move and the opponent reply. The next state is therefore the same agent's next decision. Terminal states do not bootstrap. A fixed opponent mixture is sampled independently at every opponent move: 25% random, 15% tactical, 60% minimax. The deployed agent uses only learned Q-values.

The state has nine cells: `0` empty, `1` agent, `2` opponent. Rotations and reflections share table entries during training. Roles alternate each episode. Exploration decays linearly from 1.0 to 0.03 over the first 85% of training.'''),
        nbf.v4.new_code_cell('''from tictactoe.train import TrainingConfig, run_experiment
config = TrainingConfig(episodes=160_000, seed=42, alpha=0.15, gamma=0.97,
                        final_games_per_role=2000)
ARTIFACTS = WORK / "artifacts"
print(config)
'''),
        nbf.v4.new_markdown_cell('### 3. Train and evaluate\nThe untrained baseline has all-zero Q-values and chooses uniformly among legal ties. The final evaluation uses a separate seed, 2,000 games per role against each of three opponents, and zero exploration.'),
        nbf.v4.new_code_cell('summary = run_experiment(config, ARTIFACTS, plots=True, progress=True)'),
        nbf.v4.new_markdown_cell('### 4. Inspect measured results\nWin rate and non-loss rate are different. Perfect play draws in Tic Tac Toe. The exact adversarial audit also searches for any opponent that could exploit the learned policy from an empty board, including every greedy tie.'),
        nbf.v4.new_code_cell('''evaluation = summary["evaluation"]
rows = ["| Opponent | Policy | Games | Wins | Draws | Losses | Win rate | Non-loss | Average reward |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for name in ("random", "tactical", "minimax"):
    for policy in ("baseline", "trained"):
        m = evaluation[policy][name]["combined"]
        rows.append(f"| {name} | {policy} | {m['games']} | {m['wins']} | {m['draws']} | {m['losses']} | {m['win_rate']:.2%} | {m['non_loss_rate']:.2%} | {m['average_reward']:.4f} |")
display(Markdown("\\n".join(rows)))
print("Adversarial audit:", json.dumps(evaluation["adversarial_audit"], indent=2))
print("Canonical states:", summary["canonical_states"], "Exported states:", summary["exported_states"])
'''),
        nbf.v4.new_markdown_cell('### 5. Visualize learning\nTraining rewards include exploration. Checkpoint non-loss rates use the greedy policy. A shorter episode is not automatically better; games against strong players often require all nine moves. Finite stable performance is not a proof of general Q-learning convergence with a constant learning rate.'),
        nbf.v4.new_code_cell('''for filename in ("reward_vs_episode.png", "success_vs_episode.png", "steps_vs_episode.png", "td_error_vs_episode.png"):
    display(Image(filename=str(ARTIFACTS / "plots" / filename)))
'''),
        nbf.v4.new_markdown_cell('### 6. Read the Q-learning implementation\nThe update is `Q(s,a) ← Q(s,a) + α [r + γ max_legal Q(s′,a′) − Q(s,a)]`. The actual implementation below is the one used by this experiment.'),
        nbf.v4.new_code_cell('''import inspect
from tictactoe.agent import QLearningAgent
print(inspect.getsource(QLearningAgent.update))
'''),
        nbf.v4.new_markdown_cell('### 7. Verify the rules and learner\nThese checks cover all winning lines, illegal-action masking, terminal and nonterminal updates, symmetry/export agreement, and deterministic training.'),
        nbf.v4.new_code_cell('''import subprocess
check = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                       text=True, capture_output=True)
print(check.stdout + check.stderr)
assert check.returncode == 0, "A verification check failed"
'''),
        nbf.v4.new_markdown_cell('### 8. Play the trained agent offline\nChoose X or O, compare against the random baseline, and reveal Q-values. The iframe contains the complete UI and model. You can also open the generated HTML directly in any modern browser.'),
        nbf.v4.new_code_cell('''from scripts.build_offline import build
import html
offline_path = build(WORK)
game_html = offline_path.read_text(encoding="utf-8")
display(HTML('<iframe title="Offline Q-learning Tic Tac Toe" sandbox="allow-scripts" '
             'style="width:100%;height:1100px;border:0;border-radius:16px" srcdoc="'
             + html.escape(game_html, quote=True) + '"></iframe>'))
print("Open this file directly to play:", offline_path)
'''),
        nbf.v4.new_markdown_cell('''### 9. Saved outputs and viva preparation
The local experiment folder contains `artifacts/q_table.json`, `training_summary.json`, `evaluation.json`, both CSV logs, four graphs, and `TicTacToe_Offline.html`.

**Why Q-learning?** The state space is small enough for a table; the algorithm learns action values from interaction without a transition model.

**Why exclude occupied squares?** They are not valid actions; including their values in the maximum can corrupt the update.

**Why reward a draw?** A draw is a safe outcome against perfect play. It receives 0.3, below a win's 1 and above a loss's -1.

**What does epsilon do?** It balances exploration with exploitation during training. Deployment uses epsilon 0 and random ties.

**Is minimax deployed?** No. It is a training opponent and evaluation benchmark only.

**What are the limits?** This is a finite board game, not a real-world application benchmark. One training seed is reported. Constant alpha and finite episodes do not establish the general convergence theorem. Unseen states fall back to uniform legal ties.

References: [Sutton and Barto, 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/); [Watkins and Dayan, 1992](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).
''')
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'colab': {'name': 'TicTacToe_Q_Learning.ipynb', 'provenance': []},
    })
    target = ROOT / 'notebooks/TicTacToe_Q_Learning.ipynb'
    target.parent.mkdir(exist_ok=True)
    nbf.write(notebook, target)
    print(target)


if __name__ == '__main__':
    build()
