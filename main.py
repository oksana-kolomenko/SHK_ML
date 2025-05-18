# from run_models_table_data import run_models_on_table_data
# from run_new import run
import time
import numpy as np

from csv_parser import create_patient_summaries, csv_to_kv_textfile, create_general_summaries
from dummy import print_special_tokens, print_sentence_embedding
from helpers import load_summaries
#from run_models_pca import run_pca_txt_emb
#from dummy import test
#from run_models_table_data import run_models_on_table_data
#from run_models_concatenated import run_text_concatenated
from run_models_text_emb import run_models_on_txt_emb

if __name__ == '__main__':
    csv_to_kv_textfile(input_csv_path="X.csv", output_txt_path="KV_all_summaries.txt")
    #print(len(patient_summaries))
    #print_special_tokens()
    #print_sentence_embedding(text)
    run_models_on_txt_emb()
    #run_text_concatenated()
    #run_pca_txt_emb()
    #run_pca_rte()
    #run_models_on_table_data()
