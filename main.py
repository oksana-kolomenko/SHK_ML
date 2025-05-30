# from run_models_table_data import run_models_on_table_data
# from run_new import run
import time
import numpy as np

from data_preps import create_general_summaries, write_summary
from dummy import print_special_tokens, print_sentence_embedding
from run_models_pca import run_pca_txt_emb
#from dummy import test
#from run_models_table_data import run_models_on_table_data
#from run_models_concatenated import run_text_concatenated
#from run_models_text_emb import run_models_on_txt_emb

if __name__ == '__main__':
    #run_models_on_table_data()
    #summ = create_general_summaries(tab_data="X_cybersecurity_intrusion_data.csv")
    #write_summary(file_name="cybersecurity_summaries.txt", summaries=summ)
    run_pca_txt_emb()
    #run_models_on_txt_emb()
    #run_text_concatenated()

    #run_pca_rte()

