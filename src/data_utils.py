"""Dataset loading and non-IID partitioning across clients."""
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
import torch

MAX_LEN = 128

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": input_ids.clone(),
        }

def load_raw_texts(n_samples=4000):
    """Small, fast-loading text corpus for the prototype."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 20]
    return texts[:n_samples]

def dirichlet_partition(texts, num_clients, alpha, seed=0):
    """
    Split texts across num_clients using a Dirichlet(alpha) distribution
    over a coarse 'topic' proxy (here: text length bucket, as a cheap
    non-IID proxy without needing real topic labels).
    Lower alpha -> more skewed / non-IID.
    """
    rng = np.random.default_rng(seed)
    n = len(texts)

    lengths = np.array([len(t) for t in texts])
    buckets = np.digitize(lengths, np.quantile(lengths, [0.2, 0.4, 0.6, 0.8]))
    num_buckets = buckets.max() + 1

    client_indices = [[] for _ in range(num_clients)]
    for b in range(num_buckets):
        idx_b = np.where(buckets == b)[0]
        rng.shuffle(idx_b)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(idx_b)).astype(int)[:-1]
        splits = np.split(idx_b, proportions)
        for c, split in enumerate(splits):
            client_indices[c].extend(split.tolist())

    for c in range(num_clients):
        rng.shuffle(client_indices[c])

    return client_indices

def get_client_texts(num_clients=20, alpha=0.5, seed=0):
    texts = load_raw_texts()
    indices = dirichlet_partition(texts, num_clients, alpha, seed)
    client_texts = [[texts[i] for i in idxs] for idxs in indices]
    sizes = [len(t) for t in client_texts]
    print(f"Client dataset sizes: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f}")
    return client_texts

if __name__ == "__main__":
    ct = get_client_texts(num_clients=20, alpha=0.5)
    print(f"num clients: {len(ct)}, client 0 sample: {ct[0][0][:80]}")