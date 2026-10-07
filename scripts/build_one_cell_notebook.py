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
    return source


def build():
    parts = [
        "# TIC-TAC-TOE WITH Q-LEARNING — COMPLETE, SELF-CONTAINED CODE CELL\n"
        "# Run this one cell to train, evaluate, plot, and play.\n"
        "# Requirements: Python 3.10+, NumPy, Matplotlib, IPython/Jupyter.\n",
    ]
    for filename in ("core.py", "agent.py", "evaluate.py", "train.py"):
        parts.append("\n# " + "=" * 76 + "\n# " + filename + "\n# " + "=" * 76 + "\n")
        parts.append(standalone_source(filename))

    parts.append("""
# ============================================================================
# RUN THE EXPERIMENT: edit these values if you want a shorter demonstration.
# The published report and bundled model use the defaults below.
# ============================================================================
from IPython.display import display, HTML, Image, Markdown
import html as html_module
import re

EPISODES = 160_000
SEED = 42
EVALUATION_GAMES_PER_ROLE = 2_000
OUTPUT = Path.cwd() / "one_cell_outputs"

config = TrainingConfig(
    episodes=EPISODES, seed=SEED, alpha=0.15, gamma=0.97,
    final_games_per_role=EVALUATION_GAMES_PER_ROLE,
)
summary = run_experiment(config, OUTPUT, plots=True, progress=True)

# The baseline uses an all-zero table. Results are from a separate evaluation
# seed with epsilon=0 and no Q-value updates.
evaluation = summary["evaluation"]
rows = [
    "| Opponent | Policy | Games | Wins | Draws | Losses | Win rate | Non-loss rate |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for opponent_name in ("random", "tactical", "minimax"):
    for policy_name in ("baseline", "trained"):
        metrics = evaluation[policy_name][opponent_name]["combined"]
        rows.append(
            f"| {opponent_name} | {policy_name} | {metrics['games']:,} | "
            f"{metrics['wins']:,} | {metrics['draws']:,} | {metrics['losses']:,} | "
            f"{metrics['win_rate']:.2%} | {metrics['non_loss_rate']:.2%} |"
        )
display(Markdown("## Measured evaluation\\n" + "\\n".join(rows)))
display(Markdown(
    f"**Canonical states:** {summary['canonical_states']:,}  ·  "
    f"**Exported positions:** {summary['exported_states']:,}  ·  "
    f"**Adversarial audit:** "
    f"X loss probability {evaluation['adversarial_audit']['X']['loss_probability']:.3f}, "
    f"O loss probability {evaluation['adversarial_audit']['O']['loss_probability']:.3f}"
))
for plot_name in (
    "reward_vs_episode.png", "success_vs_episode.png",
    "steps_vs_episode.png", "td_error_vs_episode.png",
):
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
                "# Tic-Tac-Toe using Q-learning — all code in one cell\n\n"
                "**Reinforcement Learning Lab CA, Semester VII.** Run the code cell below once. "
                "It contains the complete game environment, Q-learning agent, opponents, "
                "training, evaluation, plots, and playable browser UI. No repository clone, "
                "download, API key, or internet connection is required. A CPU is sufficient.\n\n"
                "The default experiment trains for 160,000 games and evaluates 2,000 games "
                "per player role per opponent. Edit `EPISODES` near the end of the cell for a "
                "shorter demonstration; its results will differ from the report. "
                "Generated files appear in `one_cell_outputs/` in the notebook's working folder.\n\n"
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
