import zmq
from config import State, Stage, MODEL_NAME, WorkerConfig, set_seed
import torch, os
from transformers import LlamaConfig, LlamaForCausalLM
from accelerate import init_empty_weights
from datetime import datetime


class Worker:
    def __init__(self):
        self.create_log()
        self.ctx = zmq.Context()

        self.pushSoc = self.ctx.socket(zmq.PUSH)
        self.pushSoc.setsockopt(zmq.LINGER, 0)
        self.pushSoc.bind(WorkerConfig.PUSH_IP)

        self.pullSoc = self.ctx.socket(zmq.PULL)
        self.pullSoc.setsockopt(zmq.LINGER, 0)
        self.pullSoc.connect(WorkerConfig.PULL_IP)

        self.worker_id = WorkerConfig.WORKER_ID

        self.config = LlamaConfig.from_pretrained(MODEL_NAME)

        full_sd = LlamaForCausalLM.from_pretrained(
            MODEL_NAME,
            config=self.config,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map="cpu",
        ).state_dict()
        target_indices = [
            f"model.layers.{i}." for i in range(*WorkerConfig.LAYER_RANGES)
        ] + ["model.rotary_emb"]
        partial_sd = {
            k: v
            for k, v in full_sd.items()
            if any(k.startswith(t) for t in target_indices)
        }
        with init_empty_weights():
            model = LlamaForCausalLM(self.config)
        model.to_empty(device="cpu")
        model.load_state_dict(partial_sd, strict=False)
        self.stage = Stage(
            model.model.layers[
                WorkerConfig.LAYER_RANGES[0] : WorkerConfig.LAYER_RANGES[1]
            ]
        )
        self.rotary_emb = model.model.rotary_emb
        self.log(f"Woker{self.worker_id} init success")

    def push(self, obj):
        self.log(
            f"Push to {'controller' if self.worker_id == 3 else 'subsequent worker'}"
        )
        self.pushSoc.send_pyobj(obj)
        self.log(
            f"Successfully pushed to {'controller' if self.worker_id == 3 else 'subsequent worker'}"
        )

    def pull(self):
        self.log(
            f"Pulling from {'controller' if self.worker_id == 1 else 'previous worker'}"
        )
        return self.pullSoc.recv_pyobj()

    def close(self):
        self.pushSoc.close()
        self.pullSoc.close()
        self.ctx.destroy()
        self.log(f"worker {self.worker_id} exiting")

    def create_log(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = f"worker{self.worker_id}_log"
        self.logfilename = f"{log_dir}/{current_time}.txt"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(self.logfilename, "w+", encoding="utf-8") as f:
            print(f"{current_time} | Log init success", file=f)

    def log(self, str):
        with open(self.logfilename, "a", encoding="utf-8") as f:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f"{current_time} | {str}", file=f)

    def run(self):
        print("Worker ready")
        while True:
            state: State = self.pull()
            self.log(
                f"Successfully received from {'controller' if self.worker_id == 1 else 'previous worker'}"
            )
            device = state.hidden_states.device  # type: ignore
            seq_len = state.hidden_states.shape[1]  # type: ignore

            cache_position = torch.arange(0, state.hidden_states.shape[1], device="cpu")  # type: ignore

            position_ids = cache_position.unsqueeze(0)

            causal_mask = torch.full(
                (seq_len, seq_len), fill_value=torch.finfo(state.hidden_states.dtype).min, device=device  # type: ignore
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask[None, None, :, :].expand(state.hidden_states.shape[0], 1, seq_len, seq_len)  # type: ignore
            position_embeddings = self.rotary_emb(state.hidden_states, position_ids)
            position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
            position_embeddings = self.rotary_emb(state.hidden_states, position_ids)

            state.set_hidden_states(
                self.stage(
                    state.hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            )

            self.push(state)


if __name__ == "__main__":
    set_seed(42)
    worker = Worker()
    worker.run()
