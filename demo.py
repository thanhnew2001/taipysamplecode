from taipy.gui import Gui
from math import cos, exp

value = 10

page = """
Markdown
# Taipy *Demo*
Value: <|{value}|text|>

<|{value}|slider|>

<|{compute_data(value)}|chart|>
"""


def compute_data(decay: int) -> list:
    return [cos(i / 6) * exp(-i * decay / 600) for i in range(100)]


Gui(page).run(use_reloader=True, port=5003)
