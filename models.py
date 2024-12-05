from transformers import AutoTokenizer, AutoModel, pipeline


def create_feature_extractor(model_name):
    """
    Creates a feature extractor pipeline for a given model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokens = to
    model = AutoModel.from_pretrained(model_name)
    # kann sein, dass man die Pipeline gar nicht benutzen kann. Dann Embeddings anders eerstellen
    return pipeline("feature-extraction", model=model, tokenizer=tokenizer)


# Clinical Longformer
feature_extractor_clinical = create_feature_extractor("yikuan8/Clinical-Longformer")
"""
# BERT
feature_extractor_bert = create_feature_extractor("google-bert/bert-base-cased")

# ELECTRA small discriminator
feature_extractor_electra_small = create_feature_extractor("google/electra-small-discriminator")

# ELECTRA base discriminator
feature_extractor_electra_base = create_feature_extractor("google/electra-base-discriminator")

# ELECTRA large discriminator
feature_extractor_electra_large = create_feature_extractor("google/electra-large-discriminator")

# SimSCE sup
feature_extractor_simsce_sup = create_feature_extractor("princeton-nlp/sup-simcse-bert-base-uncased")

# SimSCE unsup
feature_extractor_simsce_unsup = create_feature_extractor("princeton-nlp/unsup-simcse-bert-base-uncased")

# ?
# E5-SMALL-V2
feature_extractor_e5_small_v2 = create_feature_extractor("intfloat/e5-small-v2")

# E5-BASE-V2
feature_extractor_e5_base_v2 = create_feature_extractor("intfloat/e5-base-v2")

# E5-LARGE-V2
feature_extractor_e5_large_v2 = create_feature_extractor("intfloat/e5-large-v2")


#############
#### BGE ####
#############
# potion-base-2M
feature_extractor_potion_base_2M = create_feature_extractor("minishlab/potion-base-2M")

# potion-base-4M
feature_extractor_potion_base_4M = create_feature_extractor("minishlab/potion-base-4M")

# potion-base-8M
feature_extractor_potion_base_8M = create_feature_extractor("minishlab/potion-base-8M")

# bge-small-en-v1.5
feature_extractor_bge_small_en_v1_5 = create_feature_extractor("BAAI/bge-small-en-v1.5")

# GIST-small-Embedding-v0
feature_extractor_gist_small_embedding_v0 = create_feature_extractor("avsolatorio/GIST-small-Embedding-v0")

# MedEmbed-small-v0.1
# feature_extractor_medembed_small_v0_1 = create_feature_extractor("MedEmbed-small-v0.1")

# bge-base-en-v1.5
feature_extractor_bge_base_en_v1_5 = create_feature_extractor("BAAI/bge-base-en-v1.5")

# GIST-Embedding-v0
feature_extractor_gist_embedding_v0 = create_feature_extractor("avsolatorio/GIST-Embedding-v0")

# MedEmbed-base-v0.1
# feature_extractor_medembed_base_v0_1 = create_feature_extractor("MedEmbed-base-v0.1")

# bge-large-en-v1.5
feature_extractor_bge_large_en_v1_5 = create_feature_extractor("BAAI/bge-large-en-v1.5")

# GIST-large-Embedding-v0
feature_extractor_gist_large_embedding_v0 = create_feature_extractor("avsolatorio/GIST-large-Embedding-v0")

# MedEmbed-large-v0.1
# feature_extractor_medembed_large_v0_1 = create_feature_extractor("MedEmbed-large-v0.1")


#############
#### GTE ####
#############
# gte-small
feature_extractor_gte_small = create_feature_extractor("thenlper/gte-small")

# gte-base
feature_extractor_gte_base = create_feature_extractor("thenlper/gte-base")

# gte-base-en-v1.5
feature_extractor_gte_base_en_v1_5 = create_feature_extractor("Alibaba-NLP/gte-base-en-v1.5")

# gte-large
feature_extractor_gte_large = create_feature_extractor("thenlper/gte-large")

# gte-large-en-v1.5
feature_extractor_gte_large_en_v1_5 = create_feature_extractor("Alibaba-NLP/gte-large-en-v1.5")

# stella_en_400M_v5 (SotA)
feature_extractor_stella_en_400M_v5 = create_feature_extractor("dunzhang/stella_en_400M_v5")  # [CLS], [SEP]
"""