from transformers import AutoTokenizer, AutoModel
from transformers import pipeline


def test():
    model_name = "dunzhang/stella_en_400M_v5"  # or correct name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    output = pipe("This is a test sentence.")
    print(len(output), len(output[0]))
