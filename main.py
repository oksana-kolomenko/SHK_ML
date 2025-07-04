# from run_models_table_data import run_models_on_table_data
# from run_new import run
import time
import numpy as np

from data_preps import create_general_summaries, write_summary, create_patient_summaries, create_general_summaries_
from dummy import print_special_tokens, print_sentence_embedding
#from run_models_pca import run_pca_txt_emb

from run_models_pca import run_pca_txt_emb
#from dummy import test
#from run_models_table_data import run_models_on_table_data
#from run_models_concatenated import run_text_concatenated
#from run_models_text_emb import run_models_on_txt_emb

if __name__ == '__main__':
    #run_models_on_table_data()
    #summ = create_general_summaries_(tab_data="X_lung_disease_data.csv",
    #                                 output_file="_lung_disease_summaries.txt")
    """
    column_name_map = {
        "network_packet_size": "Network packet size",
        "protocol_type": "Protocol type",
        "login_attempts": "Login attempts",
        "session_duration": "Session duration",
        "encryption_used": "Encryption used",
        "ip_reputation_score": "IP reputation score",
        "failed_logins": "Failed logins",
        "browser_type": "Browser type",
        "unusual_time_access": "Unusual time access",
    }

    categorial_values = {
        "unusual_time_access": {0: "no", 1: "yes"}
    }

    summ = create_general_summaries_(tab_data="X_cybersecurity_nominal.csv",
                                     output_file="_cybersecurity_nom_summaries.txt",
                                     categorial_values=categorial_values,
                                     column_name_map=column_name_map)
    """
    #write_summary(file_name="_cybersecurity_summaries.txt", summaries=summ)
    run_pca_txt_emb()
    #run_models_on_txt_emb()
    #run_text_concatenated()

    #run_pca_rte()

