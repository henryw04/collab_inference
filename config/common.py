import os
from pathlib import Path
import torch
import random
import numpy as np

"""Default settings for inference parameters"""
MAX_NEW_TOKENS = 20
TEMPERATURE = 0.1
TOP_P = 0.9
REPETITION_PENALTY = 1.2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)

MODEL_NAME = "openlm-research/open_llama_3b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class State:
    def __init__(self, addr, prompt,
                max_new_tokens = MAX_NEW_TOKENS, 
                temperature = TEMPERATURE,
                top_p = TOP_P, 
                repetition_penalty = REPETITION_PENALTY, 
                ):
        self.addr = addr
        self.prompt = prompt
        self.remaining_tokens = max_new_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        
        self.input_ids = None
        self.hidden_states = None
    
    def __str__(self):
        return f"input_ids = {self.input_ids}, hidden_states = {self.hidden_states}"

    def set_hidden_states(self,obj):
        self.hidden_states = obj

    def set_input_ids(self,obj):
        self.input_ids = obj

    def consume_token(self):
        self.remaining_tokens-=1

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)