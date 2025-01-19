from transformers import AutoTokenizer, AutoModel, pipeline
import torch


def create_feature_extractor(model_name):
    """
    Creates a feature extractor pipeline for a given model.
    Compatible with: CL, Bert, Electra, SimSce, BGE, some GTE(thenlper), tbc
    """
    print("Starting to create a feature extractor.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokens = to
    #model = AutoModel.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda:0")
    # kann sein, dass man die Pipeline gar nicht benutzen kann. Dann Embeddings anders erstellen
    # Dynamically choose device 0 = GPU
    print("Finished creating a feature extractor.")
    return pipeline("feature-extraction", model=model, tokenizer=tokenizer, device=0)

    """device = 0 if torch.cuda.is_available() else "cpu" # sonst None

    try:
        # Initialize the pipeline with dynamic device
        return pipeline("feature-extraction", model=model, tokenizer=tokenizer, device=device)
    except Exception as e:
        print(f"Error initializing pipeline with GPU. Falling back to CPU. Error: {e}")
        # Fallback to CPU if GPU fails
        return pipeline("feature-extraction", model=model, tokenizer=tokenizer, device="cpu")
    """


def create_gte_feature_extractor(model_name):
    """
    Creates a feature extractor for a given model,
    Compatible with: some GTE (Alibaba), tbc.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def extract_features(texts):
        """
        Extracts features (embeddings) for a list of texts.

        Returns:
            A list of lists where each inner list is the token embeddings for a single input text.
            Each list has shape (seq_length, hidden_dim).
        """
        # Tokenize input texts
        batch_dict = tokenizer(
            texts,
            max_length=8192,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Move input tensors to the same device as the model
        batch_dict = {key: val.to(device) for key, val in batch_dict.items()}

        # Compute embeddings
        with torch.no_grad():
            outputs = model(**batch_dict)

        # The format of create_feature_extractor expects the embeddings as a list of lists,
        # where each list is the sequence of token embeddings for a single input text.
        # Return embeddings for each text as a list of token embeddings
        return outputs.last_hidden_state.cpu().numpy().tolist()

    return extract_features


# Clinical Longformer
feature_extractor_clinical = create_feature_extractor("yikuan8/Clinical-Longformer")

# BERT
feature_extractor_bert = create_feature_extractor("google-bert/bert-base-cased")
# feature_extractor_bert = create_feature_extractor("google-bert/distilbert-base-cased")

# ELECTRA small discriminator
feature_extractor_electra_small = create_feature_extractor("google/electra-small-discriminator")

# ELECTRA base discriminator
feature_extractor_electra_base = create_feature_extractor("google/electra-base-discriminator")

# ELECTRA large discriminator
feature_extractor_electra_large = create_feature_extractor("google/electra-large-discriminator")

"""# SimSCE sup
feature_extractor_simsce_sup = create_feature_extractor("princeton-nlp/sup-simcse-bert-base-uncased")

# SimSCE unsup
feature_extractor_simsce_unsup = create_feature_extractor("princeton-nlp/unsup-simcse-bert-base-uncased")

# ?
# E5-SMALL-V2
feature_extractor_e5_small_v2 = create_feature_extractor("intfloat/e5-small-v2")

# E5-BASE-V2
feature_extractor_e5_base_v2 = create_feature_extractor("intfloat/e5-base-v2")

# E5-LARGE-V2
feature_extractor_e5_large_v2 = create_feature_extractor("intfloat/e5-large-v2")"""


#############
#### BGE ####
#############
# potion-base-2M
# feature_extractor_potion_base_2M = create_feature_extractor("minishlab/potion-base-2M") # custom code

# potion-base-4M
# feature_extractor_potion_base_4M = create_feature_extractor("minishlab/potion-base-4M") # custom code

# potion-base-8M
# feature_extractor_potion_base_8M = create_feature_extractor("minishlab/potion-base-8M") # custom code

# bge-small-en-v1.5
feature_extractor_bge_small_en_v1_5 = create_feature_extractor("BAAI/bge-small-en-v1.5")

# GIST-small-Embedding-v0
# feature_extractor_gist_small_embedding_v0 = create_feature_extractor("avsolatorio/GIST-small-Embedding-v0") # custom code

# bge-base-en-v1.5
feature_extractor_bge_base_en_v1_5 = create_feature_extractor("BAAI/bge-base-en-v1.5")

# GIST-Embedding-v0
# feature_extractor_gist_embedding_v0 = create_feature_extractor("avsolatorio/GIST-Embedding-v0") # custom code

# bge-large-en-v1.5
feature_extractor_bge_large_en_v1_5 = create_feature_extractor("BAAI/bge-large-en-v1.5")

# GIST-large-Embedding-v0
# feature_extractor_gist_large_embedding_v0 = create_feature_extractor("avsolatorio/GIST-large-Embedding-v0") # custom code

# MedEmbed-small-v0.1
# feature_extractor_medembed_small_v0_1 = create_feature_extractor("MedEmbed-small-v0.1") # custom code

# MedEmbed-base-v0.1
# feature_extractor_medembed_base_v0_1 = create_feature_extractor("MedEmbed-base-v0.1") # custom code

# MedEmbed-large-v0.1
# feature_extractor_medembed_large_v0_1 = create_feature_extractor("MedEmbed-large-v0.1") # custom code

#############
#### GTE ####
#############
# gte-small
feature_extractor_gte_small = create_feature_extractor("thenlper/gte-small")

# gte-base
feature_extractor_gte_base = create_feature_extractor("thenlper/gte-base")

# gte-base-en-v1.5
feature_extractor_gte_base_en_v1_5 = create_gte_feature_extractor("Alibaba-NLP/gte-base-en-v1.5") # custom code

# gte-large
feature_extractor_gte_large = create_feature_extractor("thenlper/gte-large")

# gte-large-en-v1.5
feature_extractor_gte_large_en_v1_5 = create_gte_feature_extractor("Alibaba-NLP/gte-large-en-v1.5") # custom code

# stella_en_400M_v5 (SotA)
# feature_extractor_stella_en_400M_v5 = create_feature_extractor("dunzhang/stella_en_400M_v5")  # [CLS], [SEP] # custom code
