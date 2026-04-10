import torch

"""
For connecting different device, you will need to find the other device's IP 
and change the PULL_IP with that.

IP structure: "tcp://[Your IP]:[Port]"


Example WorkerConfig:
Please note that this example is used within 1 computer. 
This is not heterogeneous ready

######### Worker 1 #############
LAYER_RANGES = (0,8)
PULL_IP = "tcp://127.0.0.1:6666"
PUSH_IP = "tcp://127.0.0.1:6667" 
WORKER_ID = 1

######### Worker 2 #############
LAYER_RANGES = (8,16)
PULL_IP = "tcp://127.0.0.1:6667"
PUSH_IP = "tcp://127.0.0.1:6668"
WORKER_ID = 2

######### Worker 3 #############
LAYER_RANGES = (16,26)
PULL_IP = "tcp://127.0.0.1:6668"
PUSH_IP = "tcp://127.0.0.1:6669"
WORKER_ID = 3

"""

class WorkerConfig:
    LAYER_RANGES = (0,8) # (Starting layer, desired last layer + 1)
    PULL_IP = "tcp://127.0.0.1:6666" # Change the port if needed, no need to change the IP
    PUSH_IP = "tcp://127.0.0.1:6667" # Previous node IP + Port
    WORKER_ID = 1  # Integer, Just Name, Please be unique

class Stage(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states, attention_mask, position_ids, cache_position, position_embeddings):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        return hidden_states
    
