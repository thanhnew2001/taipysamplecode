# Create an app to upload a csv and display it in a table
from taipy.gui import Gui
import pandas as pd

data = []
data_path = ""


def data_upload(state):
    state.data = pd.read_csv(state.data_path)


page = """
<|{data_path}|file_selector|on_action=data_upload|>
<|{data}|table|>
"""

Gui(page).run()
