# from run_models_table_data import run_models_on_table_data
# from run_new import run
import time
import numpy as np
from helpers import load_summaries
from run_models_text_emb import run_models_on_txt_emb
# from run_models_concatenated import run_models_concatenated


def dummy_function():
    print("Starting dummy function...")
    for i in range(3):
        print(f"Iteration {i + 1}: Doing some work...")
        time.sleep(1)  # Simulate some processing time
    print("Dummy function completed successfully!")


# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    #patient_summaries = load_summaries()
    #print(len(patient_summaries))
     run_models_on_txt_emb()
    # run_models_on_table_data()
    # run()
    # create_summaries()
