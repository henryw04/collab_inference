import zmq
import torch
import time, os
from datetime import datetime
from config import ControllerConfig, State, MODEL_NAME, set_seed
from accelerate import init_empty_weights
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.logits_process import LogitsProcessorList


class Controller:
    def __init__(self):
        self.create_log()
        self.ctx = zmq.Context()

        # 1. Frontend API Socket (ROUTER)
        self.apiSoc = self.ctx.socket(zmq.ROUTER)
        self.apiSoc.setsockopt(zmq.LINGER, 0)
        self.apiSoc.bind(ControllerConfig.APISOC_BIND)

        # 2. Worker Push Socket (PUSH)
        self.pushSoc = self.ctx.socket(zmq.PUSH)
        self.pushSoc.setsockopt(zmq.LINGER, 0)
        self.pushSoc.bind(ControllerConfig.PUSH_BIND)

        # 3. Worker Pull Socket (PULL)
        self.pullSoc = self.ctx.socket(zmq.PULL)
        self.pullSoc.setsockopt(zmq.LINGER, 0)
        self.pullSoc.connect(ControllerConfig.PULL_CONNECT)

        # 4. Poller Setup
        self.poller = zmq.Poller()
        self.poller.register(self.apiSoc, zmq.POLLIN)
        self.poller.register(self.pullSoc, zmq.POLLIN)

        self.config = LlamaConfig.from_pretrained(MODEL_NAME)
        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

        full_sd = LlamaForCausalLM.from_pretrained(
            MODEL_NAME,
            config=self.config,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map="cpu",
        ).state_dict()

        partial_sd = {
            k: v
            for k, v in full_sd.items()
            if any(t in k for t in ["model.embed_tokens", "model.norm", "lm_head"])
        }

        with init_empty_weights():
            model = LlamaForCausalLM(self.config)

        model.to_empty(device="cpu")
        model.load_state_dict(partial_sd, strict=False)

        self.embed_tokens = model.model.embed_tokens
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.log("Controller init success")

    def getRequest(self):
        self.log("Pulling from API")
        return self.apiSoc.recv_multipart()

    def respond(self, obj: State):
        self.log("Parsing response to API")
        address = obj.addr
        text = self.tokenizer.decode(obj.input_ids[0], skip_special_tokens=True)  # type: ignore
        self.apiSoc.send_multipart([address, text.encode("utf-8")])
        self.log("Response sent to API")

    def push(self, obj):
        self.log("Push to initial worker")
        self.pushSoc.send_pyobj(obj)
        self.log("Successfully pushed to initial worker")

    def pull(self):
        self.log("Pulling from terminal worker")
        return self.pullSoc.recv_pyobj()

    def close(self):
        self.pushSoc.close()
        self.pullSoc.close()
        self.apiSoc.close()
        self.ctx.destroy()
        self.log("Controller exiting")

    def create_log(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = "controller_log"
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
        while True:
            try:
                socks = dict(self.poller.poll())
            except KeyboardInterrupt:
                break

            # Handle new request from API
            if self.apiSoc in socks:
                frame = self.getRequest()
                self.log("Successfully received from API")
                state = State(frame[0], frame[-1].decode("utf-8"))

                input_tokens = self.tokenizer(
                    state.prompt, return_tensors="pt"
                ).input_ids
                state.set_input_ids(input_tokens)
                state.set_hidden_states(self.embed_tokens(state.input_ids))
                self.push(state)

            # Handle returning state from Workers
            if self.pullSoc in socks:
                state = self.pull()
                self.log("Successfully received from terminal worker")
                processors = LogitsProcessorList(
                    [
                        RepetitionPenaltyLogitsProcessor(
                            penalty=state.repetition_penalty
                        ),
                        TemperatureLogitsWarper(temperature=state.temperature),
                        TopPLogitsWarper(top_p=state.top_p),
                    ]
                )

                state.set_hidden_states(self.norm(state.hidden_states))
                logits = self.lm_head(state.hidden_states)
                next_token_logits = logits[:, -1, :]

                for processor in processors:
                    next_token_logits = processor(state.input_ids, next_token_logits)

                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                state.set_input_ids(torch.cat([state.input_ids, next_token], dim=1))
                state.consume_token()

                # Check for stopping criteria
                eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
                if state.remaining_tokens <= 0 or (
                    eos_token_id is not None and next_token.item() == eos_token_id
                ):
                    self.respond(state)
                else:
                    # Prepare for next iteration
                    state.set_hidden_states(self.embed_tokens(state.input_ids))
                    self.push(state)

            time.sleep(0.01)

        self.close()


if __name__ == "__main__":
    set_seed(42)
    ctrl = Controller()
    ctrl.run()
