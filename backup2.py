from taipy.gui import Gui
from math import cos, exp

state = {"amp": 1, "data":[]}

def update(state):
  x = [i/10 for i in range(100)]
  y = [math.sin(i)*state.amp for i in x]  
  state.data = [{"data": y}]

page = """
Amplitude: <|{amp}|slider|>
<|Data|chart|data={data}|> 
"""

Gui(page).run(state=state)