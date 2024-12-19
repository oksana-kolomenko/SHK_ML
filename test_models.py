import torch
from transformers import AutoModel, AutoTokenizer

"""
All models that need special code to be run:

# potion-base-2M
# feature_extractor_potion_base_2M = create_feature_extractor("minishlab/potion-base-2M") # custom code

# potion-base-4M
# feature_extractor_potion_base_4M = create_feature_extractor("minishlab/potion-base-4M") # custom code

# potion-base-8M
# feature_extractor_potion_base_8M = create_feature_extractor("minishlab/potion-base-8M") # custom code

# GIST-small-Embedding-v0
# feature_extractor_gist_small_embedding_v0 = create_feature_extractor("avsolatorio/GIST-small-Embedding-v0") # custom code

# GIST-Embedding-v0
# feature_extractor_gist_embedding_v0 = create_feature_extractor("avsolatorio/GIST-Embedding-v0") # custom code

# GIST-large-Embedding-v0
# feature_extractor_gist_large_embedding_v0 = create_feature_extractor("avsolatorio/GIST-large-Embedding-v0") # custom code

# gte-base-en-v1.5
#feature_extractor_gte_base_en_v1_5 = create_feature_extractor("Alibaba-NLP/gte-base-en-v1.5") # custom code

# gte-large-en-v1.5
# feature_extractor_gte_large_en_v1_5 = create_feature_extractor("Alibaba-NLP/gte-large-en-v1.5") # custom code

# stella_en_400M_v5 (SotA)
# feature_extractor_stella_en_400M_v5 = create_feature_extractor("dunzhang/stella_en_400M_v5")  # [CLS], [SEP] # custom code

tbc
"""

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]


