from taipy.gui import Gui
import numpy as np

item1 = "None"
lov = [1, 2, 3]

page = """
<|{item1}|selector|lov={lov}|>
"""

Gui(page).run()
