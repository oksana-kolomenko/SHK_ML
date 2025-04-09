import torch
from transformers import AutoTokenizer, AutoModel, pipeline


def test():
    model_name = "dunzhang/stella_en_400M_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # move model to GPU
    model = model.to("cuda")

    pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer, device=0)
    output = pipe("This is a test sentence.")
    print("len(output): ", len(output), "len(output)[0]", len(output[0]))


def print_special_tokens():
    print("Special tokens used by the tokenizer:\n")
    model_name = "sentence-transformers/gtr-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    special_tokens = tokenizer.special_tokens_map
    for name, token in special_tokens.items():
        print(f"{name}: {token}")


from sentence_transformers import SentenceTransformer


def print_sentence_embedding(text, model_name='sentence-transformers/gtr-t5-base'):
    model = SentenceTransformer(model_name)
    embedding = model.encode(text)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
    print("\n🔹 Tokenized Text:")
    print(tokens)

    print(f"Sentence: {text}")
    print("\nSentence-level embedding (first 10 dims):")
    print(embedding[:10], "...")
    print(f"Embedding shape: {embedding.shape}")

