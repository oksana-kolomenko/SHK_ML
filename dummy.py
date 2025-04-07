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
    print(len(output), len(output[0]))
