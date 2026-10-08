"""Build the self-contained Python widget notebook from the RL source."""
from pathlib import Path
import re

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks/TicTacToe_All_In_One.ipynb"


def module_source(filename: str) -> str:
    source = (ROOT / "tictactoe" / filename).read_text(encoding="utf-8")
    source = re.sub(r"(?m)^from \.core import [^\n]*\n", "", source)
    if filename == "agent.py":
        source = source.split("    def export(", 1)[0]
    return source


DEMO = '''
# Train: try moves early, then favor learned moves.
import ipywidgets as widgets
from IPython.display import display

EPISODES = 160_000
SEED = 42
agent = QLearningAgent(alpha=0.15, gamma=0.97, seed=SEED)
opponent_rng = random.Random(SEED + 1)
opponent_weights = (0.25, 0.15, 0.60)  # random, tactical, minimax

for episode in range(1, EPISODES + 1):
    agent_mark = X if episode % 2 else O
    board = EMPTY_BOARD
    if agent_mark == O:
        board = play(board, mixture_action(board, X, opponent_rng, opponent_weights), X)
    # Epsilon falls from 1.0 to 0.03.
    fraction = min(1.0, (episode - 1) / max(1, EPISODES * 0.85))
    epsilon = 1.0 + fraction * (0.03 - 1.0)
    while True:
        state = encode_state(board, agent_mark)
        action = agent.select_action(board, agent_mark, epsilon)
        board = play(board, action, agent_mark)
        if not terminal(board):
            opponent_mark = other(agent_mark)
            reply = mixture_action(board, opponent_mark, opponent_rng, opponent_weights)
            board = play(board, reply, opponent_mark)
        done = terminal(board)
        result = winner(board)
        # Reward: win +1, draw +0.3, loss -1.
        reward = ((1.0 if result == agent_mark else -1.0) if result else 0.3) if done else 0.0
        agent.update(state, action, reward,
                     None if done else encode_state(board, agent_mark), done)
        if done:
            break


class NotebookTicTacToe:
    def __init__(self, trained_agent):
        self.agent = trained_agent
        self.board = EMPTY_BOARD
        self.human_mark = X
        self.agent_mark = O
        self.finished = False

        self.title = widgets.Label(value="TIC TAC TOE  ·  Q-LEARNING")
        self.status = widgets.Label()
        self.play_x = widgets.Button(description="Play as X", layout=widgets.Layout(width="120px"))
        self.play_o = widgets.Button(description="Play as O", layout=widgets.Layout(width="120px"))
        self.play_x.on_click(lambda _: self._change_side(X))
        self.play_o.on_click(lambda _: self._change_side(O))
        self.side = widgets.HBox([self.play_x, self.play_o])
        self.new_game = widgets.Button(description="New game", icon="refresh")
        self.new_game.on_click(lambda _: self.reset())

        self.cells = []
        for position in range(9):
            button = widgets.Button(
                description="", layout=widgets.Layout(width="82px", height="82px"),
            )
            button.style.font_weight = "bold"
            button.style.font_size = "34px"
            button.on_click(lambda _, index=position: self.human_move(index))
            self.cells.append(button)

        self.q_title = widgets.Label(value="Q-values at the AI's decision")
        self.q_state = widgets.Label(value="Make a move to see the Q-table.")
        self.q_choice = widgets.Label(value="0 = empty · 1 = AI · 2 = you")
        self.q_cells = [widgets.Button(
            description="—", layout=widgets.Layout(width="92px", height="52px"),
        ) for _ in range(9)]
        for button in self.q_cells:
            button.style.font_size = "14px"
            button.style.font_weight = "bold"
            button.tooltip = "Q value at the AI's last decision"

        board_grid = widgets.GridBox(
            self.cells,
            layout=widgets.Layout(grid_template_columns="repeat(3, 82px)", grid_gap="5px"),
        )
        q_grid = widgets.GridBox(
            self.q_cells,
            layout=widgets.Layout(grid_template_columns="repeat(3, 92px)", grid_gap="5px"),
        )
        left = widgets.VBox([self.title, self.side, self.status, board_grid, self.new_game])
        right = widgets.VBox([self.q_title, self.q_state, q_grid, self.q_choice])
        self.widget = widgets.HBox(
            [left, right],
            layout=widgets.Layout(flex_flow="row wrap", gap="32px", align_items="flex-start"),
        )
        self.reset()

    def _change_side(self, mark):
        self.human_mark = mark
        self.agent_mark = other(mark)
        self.reset()

    def _draw_side(self):
        for button, mark in ((self.play_x, X), (self.play_o, O)):
            chosen = mark == self.human_mark
            button.style.button_color = "#50d99a" if chosen else "#f8fafc"
            button.style.text_color = "#102018" if chosen else "#17212b"
            button.style.font_weight = "bold"

    def _draw_board(self):
        for index, button in enumerate(self.cells):
            mark = self.board[index]
            button.description = "X" if mark == X else "O" if mark == O else ""
            button.style.button_color = "#72e0a6" if mark == X else "#ffbe73" if mark == O else "#f8fafc"
            button.style.text_color = "#082c1b" if mark == X else "#422300" if mark == O else "#17212b"
            button.tooltip = "Empty square" if mark == 0 else "Occupied square"

    def _finish_or_continue(self):
        result = winner(self.board)
        if result or terminal(self.board):
            self.finished = True
            self.status.value = (
                "You win!" if result == self.human_mark else
                "AI wins!" if result == self.agent_mark else "Draw!"
            )
        else:
            self.status.value = "Your turn. Choose an empty square."
        self._draw_board()

    def reset(self):
        self.board = EMPTY_BOARD
        self.finished = False
        self._draw_side()
        self.q_state.value = "Make a move to see the Q-table."
        self.q_choice.value = "0 = empty · 1 = AI · 2 = you"
        for button in self.q_cells:
            button.description = "—"
            button.style.button_color = "#f8fafc"
            button.style.text_color = "#17212b"
        self.status.value = "Your turn. Choose an empty square."
        self._draw_board()
        if self.agent_mark == X:
            self._agent_move()

    def human_move(self, index):
        if self.finished or self.board[index] != 0:
            return
        self.board = play(self.board, index, self.human_mark)
        if terminal(self.board):
            self._finish_or_continue()
        else:
            self._agent_move()

    def _agent_move(self):
        # Show Q-values before the AI places its mark.
        before = self.board
        state = encode_state(before, self.agent_mark)
        values = self.agent.values(state)
        selected = self.agent.select_action(before, self.agent_mark, epsilon=0.0)
        self.q_state.value = "State: " + state + "  (AI's view)"
        for index, button in enumerate(self.q_cells):
            button.description = "occupied" if before[index] != 0 else f"{values[index]:+.3f}"
            button.style.button_color = (
                "#72e0a6" if index == selected else
                "#dbe4ea" if before[index] != 0 else "#f8fafc"
            )
            button.style.text_color = "#082c1b" if index == selected else "#243444"
        row, col = divmod(selected, 3)
        self.q_choice.value = (
            f"AI chose row {row + 1}, column {col + 1} · Q = {values[selected]:+.3f}"
        )
        self.board = play(before, selected, self.agent_mark)
        self._finish_or_continue()


game = NotebookTicTacToe(agent)
display(game.widget)
'''


def build():
    code = "\n".join([
        "# One Python cell: train the Q-learning agent, then play it.",
        module_source("core.py"),
        module_source("agent.py"),
        DEMO,
    ])
    compile(code, str(DESTINATION), "exec")
    notebook = nbf.v4.new_notebook(
        cells=[
            nbf.v4.new_markdown_cell(
                "# Tic-Tac-Toe Q-learning — Python board\n\n"
                "Run the code cell. Play on the left; see the AI's Q-values on the right. "
                "Choose X or O, or press **New game**.\n\n"
                "To explain it: **state** = board from the AI's view (0 empty, 1 AI, 2 you); "
                "**action** = empty cell; **reward** = +1 win, +0.3 draw, -1 loss; "
                "**Q-learning** updates move values after each game turn.\n\n"
                "Default: 160,000 training games. Edit `EPISODES` for a shorter run. "
                "Requires Python 3.10+, Jupyter, and `ipywidgets`."
            ),
            nbf.v4.new_code_cell(code),
        ],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"name": DESTINATION.name, "provenance": []},
        },
    )
    DESTINATION.parent.mkdir(exist_ok=True)
    nbf.write(notebook, DESTINATION)
    print(f"{DESTINATION}: one Python code cell, {len(code):,} characters")


if __name__ == "__main__":
    build()
