# from run_models_table_data import run_models_on_table_data
# from run_new import run
import time
import numpy as np

from data_preps import create_general_summaries_
#from data_preps import create_general_summaries, write_summary, create_patient_summaries, create_general_summaries_
from dummy import print_special_tokens, print_sentence_embedding
from mimic_prep import convert_and_save

#from run_models_pca import run_pca_txt_emb
#
#from run_setup import run_txt_emb
#from dummy import test
#from run_models_table_data import run_models_on_table_data

if __name__ == '__main__':
    #run_models_on_table_data()
    create_general_summaries_(tab_data=r"..\mimic\task_0_X_test.csv",
                              output_file=r"..\mimic\summaries_test_task_0.txt")

    """column_name_map = {
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
    """
    """categorial_values = {
        "unusual_time_access": {0: "no", 1: "yes"}
    }
    """
    # Example usage
    """convert_and_save(
        train_path=r"..\mimic\tasks\task_0_X_train.csv",
        test_path=r"..\mimic\tasks\task_0_X_test.csv",
        out_train_path=r"..\mimic\task_0_X_train.csv",
        out_test_path=r"..\mimic\tasks\task_0_X_test.csv",
    )"""
    #summ = create_general_summaries_(tab_data="X_lungdisease_nom.csv",
    #                                 output_file="lungdisease_nom_summaries.txt",)
                                     #categorial_values=categorial_values,
                                     #column_name_map=column_name_map)
    #run_txt_emb()
