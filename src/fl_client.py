"""Flower client: local LoRA training + Fisher diagonal (S_i) computation."""
import flwr as fl
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict

from model_utils import load_base_model, wrap_with_lora, get_lora_state_dict, set_lora_state_dict
from data_utils import TextDataset

LOCAL_EPOCHS = 1
LR = 1e-4
BATCH_SIZE = 4

def state_dict_to_ndarrays(state_dict):
    return [v.cpu().numpy() for v in state_dict.values()]

def ndarrays_to_state_dict(keys, arrays):
    return OrderedDict({k: torch.tensor(a) for k, a in zip(keys, arrays)})


class LoRAClient(fl.client.NumPyClient):
    def __init__(self, client_id, texts, tokenizer, shared_model, rank=8):
        self.client_id = client_id
        self.tokenizer = tokenizer
        self.dataset = TextDataset(texts, tokenizer)
        self.model = shared_model          # shared, not owned
        self.lora_keys = list(get_lora_state_dict(self.model).keys())
        self.last_fisher = None

    def get_parameters(self, config):
        sd = get_lora_state_dict(self.model)
        return state_dict_to_ndarrays(sd)

    def set_parameters(self, parameters):
        sd = ndarrays_to_state_dict(self.lora_keys, parameters)
        set_lora_state_dict(self.model, sd)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=LR
        )

        # accumulate squared gradients -> diagonal empirical Fisher (S_i)
        fisher_accum = {k: torch.zeros_like(v) for k, v in get_lora_state_dict(self.model).items()}
        n_batches = 0

        for _ in range(LOCAL_EPOCHS):
            for batch in loader:
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                loss.backward()

                # S_i update: elementwise square of grad, before optimizer step
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in fisher_accum and param.grad is not None:
                            fisher_accum[name] += param.grad.detach() ** 2
                n_batches += 1

                optimizer.step()

        # average Fisher estimate over batches
        self.last_fisher = {k: (v / max(n_batches, 1)) for k, v in fisher_accum.items()}

        new_sd = get_lora_state_dict(self.model)
        return (
            state_dict_to_ndarrays(new_sd),
            len(self.dataset),
            {"client_id": self.client_id},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE)
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_loss += outputs.loss.item() * batch["input_ids"].size(0)
                n += batch["input_ids"].size(0)
        avg_loss = total_loss / max(n, 1)
        return avg_loss, n, {"loss": avg_loss}

    def get_fisher(self):
        """Called by the harness after fit() to retrieve S_i for commitment."""
        return {k: v.cpu().numpy() for k, v in self.last_fisher.items()}