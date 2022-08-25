import torch as th
import sys
from calculator import sample
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def main(qn):
    device = th.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("model_ckpts")
    model.to(device)
    print("Model Loaded")

    sample_len = 100
    print(qn.strip())
    print(sample(model, qn, tokenizer, device, sample_len))


#if __name__ == "__main__":
#    main()

## Dissable below when running apiGateway.py
main(sys.argv[1])