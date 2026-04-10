import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

import time
import random
import numpy as np

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
    
MODEL_PATH = "openlm-research/open_llama_3b"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(42)

    print("Loading model... this may take a while on first run.")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )



    # test prompts
    prompts = [
        "Q: What color is the sky?\nA:",
        "Q: What is the capital of France?\nA:",
        "My name is Alice.",
        "Q: How to make a pizza?\nA:",
        "Explain why the seasons change.",
    ]

    max_new_tokens = 20
    temperature = 0.1
    top_p = 0.9
    repetition_penalty = 1.2

    print("\nRunning single-device baseline...\n")

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                repetition_penalty = repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                
            )
        end = time.time()

        elapsed = end - start
        generated_tokens = output_ids.shape[1] - input_ids.shape[1]

        # Decode only the new tokens
        text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        with open("baseline_result.txt", "a", encoding="utf-8") as f:
            print(file=f)
            print("=== Prompt ===", file=f)
            print(prompt, file=f)
            print("--- Output ---", file=f)
            print(text, file=f)
            print("=== Stat ===", file=f)
            print(f"Time={elapsed:.3f}s, Generated_tokens={generated_tokens}, Avg Tokens/s={generated_tokens/elapsed:.2f}", file=f)
            print(file=f)



if __name__ == "__main__":
    main()
