This is a python solver of the [Zebra Logic Puzzles](https://huggingface.co/spaces/allenai/ZebraLogic) using linear programming, NLP and regular expression
<img src="https://github.com/user-attachments/assets/9de6d464-c265-4aa7-a5e8-0031877e33ed" width="75%">

### Install

`pip install -r requirements.txt`

### Run

2.1. Command line version loading the Zebra Logic Puzzle descriptions in a file

`python o3_mini_solver_generic_new.py full_puzzle.txt`

2.1.1. if you want to see how the attributes and constrains are parsed use "--debug"

`python .\o3_mini_solver_generic_new.py --debug full_puzzle.txt`

2.2. Streamlit webapp version

`streamlit run .\o3_mini_solver_generic_streamlit.py`

Then use browser to open `http://localhost:8501`


