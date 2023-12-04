# Main Application
import os
import re

from taipy.gui import Gui, notify, navigate

import pandas as pd
from datetime import datetime
import chardet

from utils import (
    contains_related_word,
    categorize_columns_by_datatype,
    generate_prompts,
    all_chart_types,
)

from similar_columns import replace_values_in_string

import csv
import os

from llm_utils import (
    prompt_localllm_fsl,
    prompt_localllm_fsl_plot,
)

MAX_FILE_SIZE_MB = 22  # Maximum allowed file size in MB

LOCAL_LLM_URL = "http://20.234.124.198:5000/generate_code"

ORIGINAL_DATA_PATH = "sales_data_sample.csv"
original_data = pd.read_csv(ORIGINAL_DATA_PATH, sep=",", encoding="ISO-8859-1")
original_data["ORDERDATE"] = pd.to_datetime(original_data["ORDERDATE"])
original_data = original_data.sort_values(by="ORDERDATE")

df = pd.DataFrame(original_data)
df.columns = df.columns.str.upper()

default_data = original_data.copy()
data = df
processed_data = original_data.copy()
user_input = ""
content = None
data_path = ""
render_examples = True
show_tips = True
past_prompts = []
plot_result = ""

suggested_prompts = [""] * 5
sample_user_inputs = [
    "What are the 5 most profitable cities?",
    "Plot in a bar chart sales of the 5 most profitable cities",
    "Plot sales by product line in a pie chart",
    "Plot in a pie chart sales by country",
    "Display in a bar chart sales by product line",
]

show_suggested_prompts = False
prompt_mode = True
data_mode = False
show_modified_data = True
edit_table = pd.DataFrame()

debug_log = ""
expandPromptHelp = False

CONTEXT_PATH = "context_data.csv"
context_data = pd.read_csv(CONTEXT_PATH, sep=";")
context = ""
for instruction, code in zip(context_data["instruction"], context_data["code"]):
    example = f"{instruction}\n{code}\n"
    context += example


# Categorize columns by type for the prompt builder
categorized_columns = categorize_columns_by_datatype(df)
float_columns = categorized_columns["float_columns"]
int_columns = categorized_columns["int_columns"]
string_columns = categorized_columns["string_columns"]
date_columns = categorized_columns["date_columns"]
float_int_columns = float_columns + int_columns
date_string_columns = date_columns + string_columns
date_string_columns_toggle = date_string_columns.copy()
selected_chart_types = ""
selected_date_string_columns = ""
selected_float_int_columns = ""


def reset_prompt_builder(state) -> None:
    """
    Resets the list of possible values for the prompt builder
    """
    state.categorized_columns = categorize_columns_by_datatype(state.data)
    divide_columns(state)
    state.selected_chart_types = ""
    state.selected_date_string_columns = ""
    state.selected_float_int_columns = ""


def divide_columns(state) -> None:
    """
    Divides columns by type for the prompt builder
    """
    state.float_columns = state.categorized_columns["float_columns"]
    state.int_columns = state.categorized_columns["int_columns"]
    state.string_columns = state.categorized_columns["string_columns"]
    state.date_columns = state.categorized_columns["date_columns"]
    state.float_int_columns = state.float_columns + state.int_columns
    state.date_string_columns = state.date_columns + state.string_columns
    state.date_string_columns_toggle = state.date_string_columns.copy()


def plot(state) -> None:
    """
    Prompts local starcoder to modify or plot data

    Args:
        state (State): Taipy GUI state
    """
    state.p.update_content(state, "")

    response = prompt_localllm_fsl_plot(
        state.data.head(), state.user_input, 32, LOCAL_LLM_URL
    )

    code = re.split("\n", response[0])[0]

    code = f"<{code}"
    if not code.endswith("|>"):
        code += "|>"

    # state.plot_result = plot_prompt(API_URL, headers, context, state, state.user_input)
    output_code = replace_values_in_string(code, state.data.columns.tolist())
    state.plot_result = output_code
    print(f"Plot Code: {state.plot_result}")
    state.debug_log = state.debug_log + f"; Generated Taipy Code: {state.plot_result}"
    state.p.update_content(state, state.plot_result)
    notify(state, "success", "Plot Updated!")


def uppercase_field_labels(code):
    # Use regular expression to find text with eventual commas between [' and ']
    pattern = r"\['(.*?)'\]"
    modified_code = re.sub(pattern, lambda match: f"['{match.group(1).upper()}']", code)

    return modified_code


def modify_data(state) -> None:
    """
    Prompts local starcoder to modify or plot data
    """
    notify(state, "info", "Running query...")

    reset_data(state)

    state.content = None
    current_time = datetime.now().strftime("%H:%M")
    state.past_prompts = [current_time + "\n" + state.user_input] + state.past_prompts

    print(f"User Input: {state.user_input}")

    response = prompt_localllm_fsl(state.data, state.user_input, 64, LOCAL_LLM_URL)
    # code = re.split('|', response[0])[0]

    code = response[0].split("|")[0]
    code = uppercase_field_labels(code)

    plot_index = code.find(".plot")
    if plot_index != -1:
        code = code[:plot_index]

    # Create a dictionary for globals and locals to use in the exec() function
    globals_dict = {}
    locals_dict = {"df": state.data}  # Include 'df' if it's not already available
    # Execute the code as a string
    import_code = "import pandas as pd;"
    # If code does not start with "df = ", add it
    if not code.startswith("df = "):
        code = "df = " + code
    print(f"Data Code: {code}")
    state.debug_log = f"Generated Pandas Code: {code}"
    try:
        exec(import_code + code, globals_dict, locals_dict)
        pandas_output = locals_dict["df"]
    except Exception as e:
        on_exception(state, "modify_data", e)
        return

    # Parse if output is DataFrame, Series, string...
    if isinstance(pandas_output, pd.DataFrame):
        state.data = pandas_output
        notify(state, "success", "Data successfully modified!")
    elif isinstance(pandas_output, pd.Series):
        state.data = pd.DataFrame(pandas_output).reset_index()
        notify(state, "success", "Data successfully modified!")
    # If int, str, float, bool, list
    elif isinstance(pandas_output, (int, str, float, bool, list)):
        state.data = pd.DataFrame([pandas_output])
        notify(state, "success", "Data successfully modified!")
    # Everything else
    else:
        state.data = state.data
        state.show_modified_data = True

    # If user asked for a plot
    if contains_related_word(state.user_input):
        state.show_modified_data = True
        plot(state)


def on_exception(state, function_name: str, ex: Exception) -> None:
    """
    Catches exceptions and notifies user in Taipy GUI

    Args:
        state (State): Taipy GUI state
        function_name (str): Name of function where exception occured
        ex (Exception): Exception
    """
    notify(state, "error", f"An error occured in {function_name}: {ex}")


def reset_data(state) -> None:
    """
    Resets data to original data, resets plot
    """
    state.data = state.default_data.copy()


def example(state, id, _) -> None:
    """
    Runs an example prompt
    """
    _index = int(id.split("example")[1])
    state.user_input = state.sample_user_inputs[_index]
    modify_data(state)


def suggest_prompt(state, id, _) -> None:
    """
    Runs an suggest prompt
    """
    _index = int(id.split("suggest")[1])
    state.user_input = state.suggested_prompts[_index]
    modify_data(state)


def remove_spaces_and_convert_to_numeric(value):
    if isinstance(value, str):
        return pd.to_numeric(value.replace(" ", ""), errors="coerce")
    return value


def read_data(file_path: str):
    """
    Read csv file from a path and remove spaces from columns with numeric values

    Args:
        file_path: Path to csv file
    """

    try:
        # Check the file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # in MB
        if file_size_mb > MAX_FILE_SIZE_MB:
            print(
                f"File size exceeds {MAX_FILE_SIZE_MB}MB. Please choose a smaller file."
            )
            return "Max_File"

        # Detect the file encoding
        with open(file_path, "rb") as file:
            result = chardet.detect(file.read())
        detected_encoding = result["encoding"]

        # Detect the delimiter using csv.Sniffer
        try:
            with open(file_path, "r", encoding=detected_encoding) as file:
                sniffer = csv.Sniffer()
                sample_data = file.read(1024)  # Read a sample of the data
                delimiter = sniffer.sniff(sample_data).delimiter
        except Exception as e:
            print(f"Error detecting delimiter: {e}")
            delimiter = ","

        output_csv_file_path = "modified_file.csv"
        rows = []
        # Open the input CSV file for reading and the output CSV file for writing
        with open(file_path, "r") as input_file, open(
            output_csv_file_path, "w"
        ) as output_file:
            # Iterate through each line in the input file
            csv_reader = csv.reader(input_file)
            # Iterate through each row in the CSV file
            found_header = False
            for row in csv_reader:
                found = 0
                for cell in row:
                    if cell == "":
                        found = found + 1
                if found_header:
                    rows.append(row)
                elif found <= 2:
                    found_header = True
                    rows.append(row)

            # Specify the CSV file path where you want to save the data
            csv_writer = csv.writer(output_file)
            for row in rows:
                csv_writer.writerow(row)

        # Read the data using detected encoding and delimiter
        df = pd.read_csv(
            output_csv_file_path,
            encoding=detected_encoding,
            delimiter=delimiter,
            on_bad_lines="skip",
        )

        # Remove spaces in numeric columns
        columns_with_spaces = []
        for column in df.columns:
            if df[column].dtype == "object":  # Check if the column contains text
                if df[column].str.contains(r"\d{1,3}( \d{3})+").any():
                    columns_with_spaces.append(column)
        for column in columns_with_spaces:
            df[column] = df[column].apply(remove_spaces_and_convert_to_numeric)

        return df
    except Exception as e:
        print(f"Error reading data: {e}")
        return None


def data_upload(state) -> None:
    """
    Changes dataset to uploaded dataset
    Generate prompt suggestions
    """

    state.p.update_content(state, "")
    state.suggested_prompts = []
    state.show_tips = False

    content = read_data(state.data_path)
    if content is str:
        notify(state, "error", f"File size exceeds {MAX_FILE_SIZE_MB}MB.")
        return None

    state.default_data = content

    df = pd.DataFrame(state.default_data)
    df.columns = df.columns.str.upper()

    # get list of columns with same data types
    categorized_columns = categorize_columns_by_datatype(df)

    # prompt builder
    state.categorized_columns = categorize_columns_by_datatype(df)
    divide_columns(state)

    prompts = generate_prompts(state.categorized_columns, 5)
    state.suggested_prompts = prompts

    # Convert specified columns to datetime
    for column in categorized_columns["date_columns"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    # Convert specified columns to string
    for column in categorized_columns["string_columns"]:
        df[column] = df[column].astype("string")

    state.data = df

    state.processed_data = state.default_data.copy()

    state.render_examples = False
    state.show_suggested_prompts = True
    show_columns_fix(state)


def reset_app(state) -> None:
    """
    Resets app to original state
    """
    state.p.update_content(state, "")
    state.default_data = original_data.copy()
    reset_data(state)
    state.user_input = ""
    state.content = None
    state.data_path = ""
    state.render_examples = True
    state.show_tips = True
    state.past_prompts = []
    state.plot_result = ""
    state.suggested_prompts = [""] * 5
    state.show_suggested_prompts = False
    state.prompt_mode = True
    state.data_mode = False
    state.show_modified_data = True
    state.edit_table = pd.DataFrame()
    state.processed_data = original_data.copy()
    show_columns_fix(state)
    reset_prompt_builder(state)
    state.categorized_columns = categorize_columns_by_datatype(df)
    navigate(state, force=True)


def show_prompt(state, id, action) -> None:
    """
    Selects the active page between "Prompt" and "Data Processing"
    """
    show_columns_fix(state)
    if "show_prompt_button" in id:
        state.prompt_mode = True
        state.data_mode = False
    if "show_data_processing_button" in id:
        state.prompt_mode = False
        state.data_mode = True


def show_columns_fix(state):
    """
    On Data Processing Page, generate the title and data type text fields
    """
    # Get the titles and data types from the header
    try:
        df = pd.DataFrame(state.processed_data)
        title_row = df.columns.tolist()
        data_types = df.dtypes.tolist()

        state.edit_table = pd.DataFrame(
            [title_row, [reverse_types_dict[str(d)] for d in data_types]],
            columns=title_row,
        )

        state.partial_columns_fix.update_content(
            state,
            """<|{edit_table}|table|show_all|on_edit=on_edit|width=100%|class_name=edit_table|>
            *Accepted values for types are: int, float, str, date, bool*{: .text-small}
            """,
        )

        categorized_columns = categorize_columns_by_datatype(state.data)
        prompts = generate_prompts(categorized_columns, 5)
        state.suggested_prompts = prompts

    except Exception as e:
        print(f"Error reading data: {e}")
        return None


def on_edit(state, var_name, action, payload):
    index = payload["index"]
    col = payload["col"]
    value = payload["value"]

    col = state.edit_table.columns.get_loc(col)

    if index == 0:
        on_title_change(state, index, col, value)
    elif index == 1:
        on_datatype_change(state, index, col, value)

    more_prompt(state)
    reset_prompt_builder(state)
    state.default_data = state.data.copy()


def on_title_change(state, index, col, value):
    """
    Changes the title of a column as requested by the user
    """
    df = pd.DataFrame(state.processed_data)
    df.rename(columns={df.columns[col]: value}, inplace=True)
    state.data = state.processed_data.copy()
    show_columns_fix(state)


types_dict = {
    "int": "int64",
    "float": "float64",
    "str": "string",
    "date": "datetime64[ns]",
    "bool": "bool",
}

reverse_types_dict = {
    "int64": "int",
    "float64": "float",
    "string": "str",
    "datetime64[ns]": "date",
    "bool": "bool",
    "object": "object",
}


def on_datatype_change(state, index, col, value):
    """
    Changes the data type of a column as requested by the user
    """
    # Check if value is in types_dict
    if value not in types_dict:
        notify(
            state, "error", "The only accepted values are: int, float, str, date, bool"
        )
        return
    value = types_dict[value]
    df = pd.DataFrame(state.processed_data)
    if value in ["int64", "float64"]:
        notify(state, "info", "Non-numeric values will be removed")
        df.iloc[:, col] = pd.to_numeric(df.iloc[:, col], errors="coerce")
        df = df.dropna()
    df.iloc[:, col] = df.iloc[:, col].astype(value)
    state.data = state.processed_data.copy()
    show_columns_fix(state)


def more_prompt(state) -> None:
    """
    Generates more prompt suggestions
    """
    df = pd.DataFrame(state.processed_data)
    categorized_columns = categorize_columns_by_datatype(df)
    prompts = generate_prompts(categorized_columns, 5)
    state.suggested_prompts = prompts


def build_prompt(state) -> None:
    """
    Generates a prompt using the prompt builder
    """
    if state.selected_date_string_columns != "":
        state.user_input = f"Plot a {state.selected_chart_types} of {state.selected_float_int_columns} by {state.selected_date_string_columns}"
    else:
        state.user_input = (
            f"Plot a {state.selected_chart_types} of {state.selected_float_int_columns}"
        )
    modify_data(state)


def on_select_change(state) -> None:
    """
    Restricts the possible values for the prompt builder according to datatype
    """
    if state.selected_chart_types == "histogram":
        state.date_string_columns_toggle = []
        state.selected_date_string_columns = ""
    elif state.selected_chart_types == "scatter plot":
        state.date_string_columns_toggle = (
            state.date_string_columns + state.float_int_columns
        )
    else:
        state.date_string_columns_toggle = state.date_string_columns


page = """
<|layout|columns=300px 1|

<|part|render=True|class_name=sidebar|
# Talk To **Taipy**{: .color-primary} # {: .logo-text}

<|Reset App|button|on_action=reset_app|class_name=fullwidth plain|id=reset_app_button|>

### Previous activities ### {: .h5 .mt2 .mb-half}
<|tree|lov={past_prompts[:5]}|class_name=past_prompts_list|multiple|>

|>

<|part|render=True|class_name=p2|

<|part|class_name=tabs pl1 pr1|
<|part|render={prompt_mode}|
<|Prompt|button|on_action=show_prompt|id=show_prompt_button|class_name=tab active|>
<|Data Preprocessing|button|on_action=show_prompt|id=show_data_processing_button|class_name=tab|>
|>
<|part|render={data_mode}|
<|Prompt|button|on_action=show_prompt|id=show_prompt_button|class_name=tab|>
<|Data Preprocessing|button|on_action=show_prompt|id=show_data_processing_button|class_name=tab active|>
|>
|>

<|part|render={prompt_mode}|

<|card

### Prompt ### {: .h4 .mt0 .mb-half}
<|{user_input}|input|on_action=modify_data|class_name=fullwidth|label=Enter your prompt here|id=prompt|change_delay=550|>

<|Need help for building a prompt?|expandable|expanded={expandPromptHelp}|class_name=prompt-help mt0|

#### Prompt suggestions #### {: .h6 .mt1 .mb-half}
<|part|render={show_tips}|
<|{sample_user_inputs[0]}|button|on_action=example|class_name=button_link|id=example0|>
<|{sample_user_inputs[1]}|button|on_action=example|class_name=button_link|id=example1|>
<|{sample_user_inputs[2]}|button|on_action=example|class_name=button_link|id=example2|>
<|{sample_user_inputs[3]}|button|on_action=example|class_name=button_link|id=example3|>
<|{sample_user_inputs[4]}|button|on_action=example|class_name=button_link|id=example4|>
|>

<|part|render={show_suggested_prompts}|
<|{suggested_prompts[0]}|button|on_action=suggest_prompt|class_name=button_link|id=suggest0|>
<|{suggested_prompts[1]}|button|on_action=suggest_prompt|class_name=button_link|id=suggest1|>
<|{suggested_prompts[2]}|button|on_action=suggest_prompt|class_name=button_link|id=suggest2|>
<|{suggested_prompts[3]}|button|on_action=suggest_prompt|class_name=button_link|id=suggest3|>
<|{suggested_prompts[4]}|button|on_action=suggest_prompt|class_name=button_link|id=suggest4|>
<|More prompts|button|on_action=more_prompt|id=more_prompt_button|>
|>

#### Prompt builder ### {: .h6 .mt1 .mb-half}
<|layout|columns=auto 1 auto 1 auto 1 auto|class_name=align-columns-center
<|
Plot a
|>
<|{selected_chart_types}|selector|lov={all_chart_types}|dropdown=True|on_change=on_select_change|class_name=fullwidth|id=chart_type_select|>
<|
of
|>
<|{selected_float_int_columns}|selector|lov={float_int_columns}|dropdown=True|on_change=on_select_change|class_name=fullwidth|id=float_int_select|>
<|
by
|>
<|{selected_date_string_columns}|selector|lov={date_string_columns_toggle}|dropdown=True|on_change=on_select_change|class_name=fullwidth|id=date_string_select|>

<|Build|button|on_action=build_prompt|class_name=button_link|class_name=plain|>
|>

|>

|>

<|part|class_name=card mt1|

<|part|render=False|
### Original Data Table ### {: .h4 .mt0 .mb-half}
<|{original_data}|table|width=100%|page_size=5|rebuild|class_name=table|>
<center>
<|{content}|image|width=50%|>
</center>
|>

<|part|render={show_modified_data}|
<|Original Data Table|expandable|expanded=False|
<|{default_data}|table|width=100%|page_size=5|rebuild|class_name=table|>
|>
<br />
### Modified Data Table ### {: .h5 .mt0 .mb-half}
<|{data}|table|width=100%|page_size=5|rebuild|class_name=table|>
|>

### Graphs/Charts ### {: .h5 .mt1 .mb-half}
<|part|partial={p}|>
|>

<|Debug Logs|expandable|expanded=True|
<|{debug_log}|text|>
|>

|>

<|part|render={data_mode}|

<|card

<|layout|columns=1 auto|class_name=align-columns-center
### Data Preprocessing ### {: .h4 .mt0 .mb-half}
<|{data_path}|file_selector|on_action=data_upload|label=Upload your CSV file|class_name=plain|>
|>

#### Edit column names and data types ### {: .h6 .mt1 .mb-half}
<|part|partial={partial_columns_fix}|>
|>


<|part|class_name=card mt1|
### Data Table ### {: .h4 .mt0 .mb-half}
<|{data}|table|width=100%|page_size=5|rebuild|>
|> 

|>
<br />
Any issues or suggestions? Mail them to: **support@taipy.io**{: .color-primary}

We only store the prompts you enter for the sole purpose of improving our product and counting daily active users. We do not store any of your data. For more information, please read our [Privacy Policy](https://www.taipy.io/privacy-policy/)
|>
|>
"""
gui = Gui(page)
partial_columns_fix = gui.add_partial("")
p = gui.add_partial("")
gui.run(title="Talk To Taipy", margin="0rem", debug=True, use_reloader=True, port=5039)
