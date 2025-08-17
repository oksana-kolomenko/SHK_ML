# from run_models_table_data import run_models_on_table_data
# from run_new import run
import time
import numpy as np
import pandas as pd

from data_prep import do_add_prep, create_mimic_summaries
#from data_preps import create_general_summaries_
#from data_preps import create_general_summaries, write_summary, create_patient_summaries, create_general_summaries_
from dummy import print_special_tokens, print_sentence_embedding
#from data_prep import mimic_convert_binary_to_bool, mimic_generate_tasks, print_csv_file_lengths, \
#    mimic_add_preprocessing, do_subsample, do_add_prep, create_mimic_summaries

from run_setup import run_txt_emb
#from dummy import test
#from run_models_table_data import run_models_on_table_data

if __name__ == '__main__':
    run_txt_emb()
    # MIMIC
    # 1. Create tasks
    # mimic_generate_tasks
    # 2. Subsample
    # mimic_subsample(X_train=r"..\mimic\tasks\task_2_X_train.csv", y_labels_train=r"..\mimic\tasks\task_2_y_train.csv", task_name="task_2")
    # 3. Replace 1/0 and True/False with yes/no
    #do_add_prep()
    # 4. Create Summaries
    #create_mimic_summaries()
    # print_csv_file_lengths("..\mimic\subsampled")
