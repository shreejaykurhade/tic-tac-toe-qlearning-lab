"""Generate a single-cell notebook from the actual project implementation.

This avoids a second, drifting copy of the learner while making the notebook
entirely self-contained: it does not clone, download, or import this repository.
"""
from pathlib import Path
import re
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks/TicTacToe_All_In_One.ipynb"


def standalone_source(name: str) -> str:
    source = (ROOT / "tictactoe" / name).read_text(encoding="utf-8")
    # The four package modules are pasted into one namespace in dependency order.
    source = re.sub(r"(?m)^\s*from \.(?:core|agent|evaluate) import [^\n]*\n", "\n", source)
    if name == "train.py":
        source = source.split('\nif __name__ == "__main__":', 1)[0]
        replacements = {
            "def train(config: TrainingConfig, progress: bool = True):":
                "def train(config: TrainingConfig, progress: bool = True, snapshot_callback=None):",
            "        total_reward += reward\n":
                "        total_reward += reward\n"
                "        if snapshot_callback and (episode % 100 == 0 or episode == config.episodes):\n"
                "            snapshot_callback(episode, agent, epsilon)\n",
            "                   progress: bool = True) -> dict:":
                "                   progress: bool = True, snapshot_callback=None) -> dict:",
            "    agent, history, checkpoints, elapsed = train(config, progress)":
                "    agent, history, checkpoints, elapsed = train(config, progress, snapshot_callback)",
        }
        for old, new in replacements.items():
            if source.count(old) != 1:
                raise ValueError(f"Expected one training insertion point: {old!r}")
            source = source.replace(old, new)
    return source


def build():
    parts = [
        "# TIC-TAC-TOE WITH Q-LEARNING — ONE CODE CELL\n"
        "# Edit these three settings, then run this cell once.\n"
        "EPISODES = 160_000\n"
        "SEED = 42\n"
        "EVALUATION_GAMES_PER_ROLE = 2_000\n"
        "# Full Q-tables are saved every 100 episodes. CPU is sufficient.\n",
    ]
    for filename in ("core.py", "agent.py", "evaluate.py", "train.py"):
        parts.append("\n# " + "=" * 76 + "\n# " + filename + "\n# " + "=" * 76 + "\n")
        parts.append(standalone_source(filename))

    parts.append("""
# ============================================================================
# RUN THE EXPERIMENT using the settings at the top of this cell.
# ============================================================================
from IPython.display import display, HTML, Image, Markdown
import html as html_module
import re
import gzip

OUTPUT = Path.cwd() / "one_cell_outputs"

config = TrainingConfig(
    episodes=EPISODES, seed=SEED, alpha=0.15, gamma=0.97,
    final_games_per_role=EVALUATION_GAMES_PER_ROLE,
)
OUTPUT.mkdir(parents=True, exist_ok=True)
snapshot_path = OUTPUT / "q_table_every_100.jsonl.gz"
snapshot_index = []

with gzip.open(snapshot_path, "wt", encoding="utf-8") as snapshot_file:
    def save_q_table(episode, agent, epsilon):
        # Full canonical Q-table at this exact training episode.
        snapshot_file.write(json.dumps({
            "episode": episode,
            "epsilon": epsilon,
            "q_table": agent.q,
        }, separators=(",", ":")) + "\\n")
        snapshot_index.append({
            "episode": episode,
            "states": len(agent.q),
            "empty_board_q": agent.values("000000000"),
        })

    summary = run_experiment(
        config, OUTPUT, plots=True, progress=False,
        snapshot_callback=save_q_table,
    )

with (OUTPUT / "q_table_index_every_100.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["episode", "canonical_states"] + [f"empty_board_q{i}" for i in range(9)])
    for item in snapshot_index:
        writer.writerow([item["episode"], item["states"], *item["empty_board_q"]])

def show_q_table(episode, limit=12):
    # Display the saved table for one checkpoint, optionally limiting rows.
    if episode < 100 or episode > EPISODES or (episode % 100 and episode != EPISODES):
        raise ValueError("Choose 100, 200, ..., EPISODES")
    record = None
    with gzip.open(snapshot_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate["episode"] == episode:
                record = candidate
                break
    if record is None:
        raise ValueError("No Q-table saved for that episode")
    states = sorted(record["q_table"].items())
    selected = states if limit is None else states[:limit]
    columns = "| State (0 empty, 1 self, 2 opponent) | " + " | ".join(f"Q{i}" for i in range(9)) + " |"
    separator = "|---|" + "---:|" * 9
    lines = [columns, separator]
    lines.extend("| `" + state + "` | " + " | ".join(f"{value:.3f}" for value in values) + " |"
                 for state, values in selected)
    display(Markdown(f"### Q-table after {episode:,} games — {len(states):,} states\\n"
                     + "\\n".join(lines)))
    if limit is not None and len(states) > limit:
        print(f"Showing {limit} of {len(states)} states. Use show_q_table({episode}, limit=None) for all.")

display(Markdown(
    f"## Q-table every 100 games\\n"
    f"Saved **{len(snapshot_index):,} full Q-table snapshots** to "
    f"`{snapshot_path.name}`. The companion CSV tracks every checkpoint. "
    "Call `show_q_table(100)` or `show_q_table(1000, limit=None)` to inspect any saved table."
))
show_q_table(100, limit=8)
show_q_table(EPISODES, limit=8)

evaluation = summary["evaluation"]
rows = [
    "| Opponent | Games | Wins | Draws | Losses | Non-loss rate |",
    "|---|---:|---:|---:|---:|---:|",
]
for opponent_name in ("random", "tactical", "minimax"):
    metrics = evaluation["trained"][opponent_name]["combined"]
    rows.append(
        f"| {opponent_name} | {metrics['games']:,} | "
        f"{metrics['wins']:,} | {metrics['draws']:,} | {metrics['losses']:,} | "
        f"{metrics['non_loss_rate']:.2%} |"
    )
display(Markdown("## Measured evaluation\\n" + "\\n".join(rows)))
display(Markdown(
    f"**Canonical states:** {summary['canonical_states']:,}  ·  "
    f"**Exported positions:** {summary['exported_states']:,}  ·  "
    f"**Adversarial audit:** "
    f"X loss probability {evaluation['adversarial_audit']['X']['loss_probability']:.3f}, "
    f"O loss probability {evaluation['adversarial_audit']['O']['loss_probability']:.3f}"
))
for plot_name in ("reward_vs_episode.png", "success_vs_episode.png"):
    display(Image(filename=str(OUTPUT / "plots" / plot_name)))

# Build the actual offline game from this run's Q-table. The page contains all
# CSS, JavaScript, and learned values; opening it requires only a browser.
""")

    frontend = {
        name: (ROOT / name).read_text(encoding="utf-8")
        for name in ("index.html", "styles.css", "app.js", "assets/favicon.svg")
    }
    parts.append("FRONTEND_SOURCE = " + repr(frontend) + "\n")
    parts.append("""
import base64
page = FRONTEND_SOURCE["index.html"]
css = FRONTEND_SOURCE["styles.css"]
javascript = FRONTEND_SOURCE["app.js"]
favicon = base64.b64encode(FRONTEND_SOURCE["assets/favicon.svg"].encode()).decode()
page = page.replace('href="assets/favicon.svg"',
                    'href="data:image/svg+xml;base64,' + favicon + '"')
page = page.replace('<link rel="stylesheet" href="styles.css">',
                    '<style>\\n' + css + '\\n</style>')
page = re.sub(r'\\s*<script[^>]+src="artifacts/model.js"[^>]*></script>', '', page)
page = re.sub(r'\\s*<script[^>]+src="app.js"[^>]*></script>', '', page)
page = page.replace('href="./"', 'href="#play"')
model_json = (OUTPUT / "q_table.json").read_text(encoding="utf-8")
embedded_script = "window.TICTACTOE_MODEL=" + model_json + ";\\n" + javascript
page = page.replace("</body>",
                    "<script>\\n" + embedded_script.replace("</script", "<\\\\/script")
                    + "\\n</script>\\n</body>")
offline_game = OUTPUT / "TicTacToe_Offline.html"
offline_game.write_text(page, encoding="utf-8")
display(Markdown("## Play the trained agent\\nChoose X or O in the game below. "
                 "Use the Q-value switch to inspect decisions. "
                 f"The standalone file is saved at `{offline_game}`."))
display(HTML(
    '<iframe title="Q-learning Tic-Tac-Toe" sandbox="allow-scripts" '
    'style="width:100%;height:1100px;border:0;border-radius:16px" srcdoc="'
    + html_module.escape(page, quote=True) + '"></iframe>'
))
print("All local results:", OUTPUT)
""")

    cell_source = "\n".join(parts)
    compile(cell_source, str(DESTINATION), "exec")
    notebook = nbf.v4.new_notebook(
        cells=[
            nbf.v4.new_markdown_cell(
                "# Tic-Tac-Toe Q-learning — one cell\n\n"
                "Run the code cell once to train the agent and play it in the same interface "
                "as the offline game. Change `EPISODES` at the **top of the code cell** if you want "
                "a shorter run. The default is 160,000 episodes.\n\n"
                "A **full Q-table snapshot is saved every 100 training episodes** in "
                "`one_cell_outputs/q_table_every_100.jsonl.gz`. The companion CSV lists every "
                "checkpoint and the empty-board Q-values. The notebook shows the first and last "
                "tables; call `show_q_table(episode, limit=None)` to view any complete checkpoint. "
                "The interface and final model are bundled into `one_cell_outputs/TicTacToe_Offline.html`. "
                "Python 3.10+, NumPy, Matplotlib, and Jupyter are required. No download or API key is used.\n\n"
                "Student name: __________  ·  Roll number: __________  ·  "
                "Division/batch: __________  ·  Date: __________"
            ),
            nbf.v4.new_code_cell(cell_source),
        ],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"name": DESTINATION.name, "provenance": []},
        },
    )
    DESTINATION.parent.mkdir(exist_ok=True)
    nbf.write(notebook, DESTINATION)
    print(f"{DESTINATION} — {len(notebook.cells)} cells, 1 code cell, {len(cell_source):,} source characters")


if __name__ == "__main__":
    build()
