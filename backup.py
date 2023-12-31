from taipy.gui import Gui
from math import cos, exp

value = 10

page = """
Markdown
# Taipy *Demo*

Value: <|{value}|text|>

<|{value}|slider|on_change=on_slider|>

<|{data}|chart|>
"""

def compute_data(decay:int)->list:
    return [cos(i/6) * exp(-i*decay/600) for i in range(100)]

def on_slider(state):
    state.data = compute_data(state.value)

data = compute_data(value)

Gui(page).run(use_reloader=True, port=5002)