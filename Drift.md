<|layout|columns=1 1|
<|part|class_name=card|
### Select Reference Data<br/>
<|{ref_selected}|selector|lov=data_ref;data_noisy;data_female;data_big|dropdown|on_change=on_ref_change|>
|>

<|part|class_name=card|
### Select Comparison Data<br/>
<|{compare_selected}|selector|lov=data_ref;data_noisy;data_female;data_big|dropdown|on_change=on_compare_change|>
|>


|>

<|Reference Dataset and Compare Dataset|expandable|expanded=True|
Display ref_data and compare_data
<|layout|columns=1 1|
<|{ref_data}|table|page_size=5|>

<|{compare_data}|table|page_size=5|>
|>
|>

<|layout|columns=1 1|
<|part|class_name=card|
<|{sex_data}|chart|type=bar|x=Dataset|y[1]=Male|y[2]=Female|title=Sex Distribution|>
|>

<|part|class_name=card|
<|{bp_data}|chart|type=histogram|options={bp_options}|layout={bp_layout}|>
|>
|>

<br/>
### Run the scenario:
<|{scenario}|scenario|on_submission_change=on_submission_status_change|expandable=False|expanded=False|>

<|{scenario}|scenario_dag|>

<br/>
### View the results:
<|{scenario.drift_results if scenario else None}|data_node|>