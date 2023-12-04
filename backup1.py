from taipy.gui import Gui
from math import sin, cos, pi

state = {
  "frequency": 1,
  "decay": 0.01,
  "data": []  
}

page = """
# Sine and Cosine Functions

Frequency: <|{frequency}|slider|min=0|max=10|step=0.1|on_change=update|> 
Decay: <|{decay}|slider|min=0|max=1|step=0.01|on_change=update|>

<|Data|chart|data={data}|>
"""

def update(state):
  x = [i/10 for i in range(100)]
  
  y1 = [sin(i*state.frequency*2*pi) * exp(-i*state.decay) for i in x]
  y2 = [cos(i*state.frequency*2*pi) * exp(-i*state.decay) for i in x]  

  state.data = [
    {"name": "Sine", "data": y1},
    {"name": "Cosine", "data": y2}
  ]

Gui(page).run(use_reloader=True, state=state)