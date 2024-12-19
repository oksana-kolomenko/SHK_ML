import torch
from transformers import AutoTokenizer, AutoModel, pipeline


models = [
    # Clinical Longformer [CLS], [SEP]
    "yikuan8/Clinical-Longformer",

    # BERT [CLS], [SEP]
    "google-bert/bert-base-cased",

    # ELECTRA [CLS], [SEP]
    "google/electra-small-discriminator",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",

    # SimSCE [CLS], [SEP]
    "princeton-nlp/sup-simcse-bert-base-uncased",
    "princeton-nlp/unsup-simcse-bert-base-uncased",

    # E5 [CLS], [SEP]
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",


    # BGE
    "minishlab/potion-base-2M",  # [CLS], [SEP]
    "minishlab/potion-base-4M",  # [CLS], [SEP]
    "minishlab/potion-base-8M",  # [CLS], [SEP]
    "BAAI/bge-small-en-v1.5",  # [CLS], [SEP]
    "BAAI/bge-base-en-v1.5",  # [CLS], [SEP]
    "BAAI/bge-large-en-v1.5",  # [CLS], [SEP]
    "avsolatorio/GIST-small-Embedding-v0",  # [CLS], [SEP]
    "avsolatorio/GIST-Embedding-v0",  # [CLS], [SEP]
    "avsolatorio/GIST-large-Embedding-v0",  # [CLS], [SEP]
    #"MedEmbed-small-v0.1",
    #"MedEmbed-Small-v1",
    #"medical-bge-large-mix2",
    #"medical-bge-base-v0-mix2",
    #"medical-bge-small-v1-mix1",
    #"MedEmbed-base-v0.1",
    #"MedEmbed-large-v0.1",

    # GTE # [CLS], [SEP]
    "thenlper/gte-small",
    "thenlper/gte-base",
    "Alibaba-NLP/gte-base-en-v1.5",
    "thenlper/gte-large",
    "Alibaba-NLP/gte-large-en-v1.5",

    # Stella
    "dunzhang/stella_en_400M_v5"  # [CLS], [SEP]
]


def blub(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    text = "Hello, how are you? Are you good?"
    tokens = tokenizer.encode(text)
    print(f"\n\nSpecial tokens in {model}: {tokenizer.special_tokens_map}")
    print("Tokenized text:")
    for tok in tokens:
        print(tokenizer.decode(tok))


def blub_medEmbed(model_name):
    token = "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"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    med_model = AutoModel.from_pretrained(model_name, use_auth_token=token)

    # Example text
    text = "The patient shows symptoms of acute bronchitis."

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Forward pass through the model to get embeddings
    with torch.no_grad():
        outputs = med_model(**inputs)
        # The last hidden states are often used for embeddings
        embeddings = outputs.last_hidden_state

    # Process the embeddings
    # Example: Use the [CLS] token representation for sentence-level embedding
    sentence_embedding = embeddings[:, 0, :]

    # Convert to numpy for further processing (optional)
    sentence_embedding_np = sentence_embedding.numpy()

    print("Sentence Embedding Shape:", sentence_embedding_np.shape)


for model in models:
    blub_medEmbed(model)
