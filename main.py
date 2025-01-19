# from run_models_table_data import run_models_on_table_data
# from run_new import run
import time

import numpy as np

#from run_models_concatenated import run_models_concatenated


def dummy_function():
    print("Starting dummy function...")
    for i in range(3):
        print(f"Iteration {i + 1}: Doing some work...")
        time.sleep(1)  # Simulate some processing time
    print("Dummy function completed successfully!")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #run_models_concatenated()
    import torch

    print(f"Is CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    # run_models_on_table_data()
    # run()
    # create_summaries()
