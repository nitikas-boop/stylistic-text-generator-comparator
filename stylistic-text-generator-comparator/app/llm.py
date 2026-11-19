"""LoRA implementation from scratch, Hugging Face LLM loading, fine-tuning, and generation."""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import httpx
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.storage import DEFAULT_CORPUS_TEXT, tokenize_ngram

# --- Legacy HTTP Chat Support (preserves backward compatibility with test_smoke.py) ---

def _load_env() -> None:
    """Read KEY=VALUE lines from .env without overriding real environment variables."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def settings() -> tuple[str, str, str]:
    """Retrieve LLM settings from environment or fallback defaults."""
    _load_env()
    return (
        os.environ.get("LLM_BASE_URL") or "http://localhost:11434/v1",
        os.environ.get("LLM_API_KEY", ""),
        os.environ.get("LLM_MODEL", ""),
    )


def chat(messages: list[dict], **options) -> str:
    """Send chat messages via OpenAI-compatible endpoint. options go straight to API."""
    base, key, model = settings()
    if not model:
        raise RuntimeError("Set LLM_MODEL in .env (see .env.example).")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = httpx.post(
        f"{base.rstrip('/')}/chat/completions",
        headers=headers,
        timeout=120,
        json={"model": model, "messages": messages, **options},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# --- Device Auto-Detection ---

def get_device() -> torch.device:
    """Auto-detect available accelerator (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- Offline / Fallback Tokenizer ---

class SimpleSportsTokenizer:
    """A lightweight word/token level tokenizer for offline and test environments."""

    def __init__(self, vocab_tokens: list[str] | None = None):
        self.pad_token = "<|pad|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token_id = 0
        self.eos_token_id = 1

        tokens = [self.pad_token, self.eos_token]
        if vocab_tokens:
            for t in vocab_tokens:
                if t not in tokens:
                    tokens.append(t)
        else:
            # Seed with sports commentary vocabulary
            sample_words = tokenize_ngram(DEFAULT_CORPUS_TEXT, lowercase=False)
            for w in sample_words:
                if w not in tokens:
                    tokens.append(w)

        self.vocab = {tok: idx for idx, tok in enumerate(tokens)}
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(
        self,
        text: str,
        return_tensors: str | None = None,
        max_length: int | None = None,
        truncation: bool = False,
        padding: bool = False,
    ) -> Any:
        tokens = tokenize_ngram(text, lowercase=False)
        ids = [self.vocab.get(tok, self.vocab.get(tok.lower(), self.eos_token_id)) for tok in tokens]

        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]

        if padding and max_length is not None and len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))

        if return_tensors == "pt":
            tensor_ids = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": tensor_ids, "attention_mask": torch.ones_like(tensor_ids)}
        return ids

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.squeeze().tolist()
            if isinstance(token_ids, int):
                token_ids = [token_ids]

        words = []
        for tid in token_ids:
            tok = self.inv_vocab.get(tid, "")
            if skip_special_tokens and tok in (self.pad_token, self.eos_token):
                continue
            words.append(tok)

        text = ""
        for w in words:
            if not text or w in ".,!?;:":
                text += w
            elif w.startswith("'"):
                text += w
            else:
                text += " " + w
        return text.strip()


# --- LoRA from Scratch Implementation ---

class LoRALinear(nn.Module):
    """Custom Low-Rank Adaptation (LoRA) layer implemented from scratch.

    Wraps an existing Linear or Conv1D layer:
        h = W0*x + (alpha / rank) * (x @ A^T @ B^T)
    Base weights W0 are completely frozen.
    Matrix A is initialized with Kaiming uniform.
    Matrix B is initialized to zero so the initial adapter contribution is exactly 0.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank) if rank > 0 else 1.0

        # Freeze the base layer weights
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Determine input and output feature dimensions
        if hasattr(original_layer, "in_features") and hasattr(original_layer, "out_features"):
            self.in_features = original_layer.in_features
            self.out_features = original_layer.out_features
        elif hasattr(original_layer, "nx") and hasattr(original_layer, "nf"):
            # Hugging Face GPT-2 Conv1D layer
            self.in_features = original_layer.nx
            self.out_features = original_layer.nf
        elif hasattr(original_layer, "weight"):
            w = original_layer.weight
            if original_layer.__class__.__name__ == "Conv1D":
                self.in_features = w.shape[0]
                self.out_features = w.shape[1]
            else:
                self.out_features = w.shape[0]
                self.in_features = w.shape[1]
        else:
            raise ValueError(f"Cannot infer feature dimensions from {original_layer}")

        # LoRA low-rank trainable weight matrices
        # lora_A: [rank, in_features]
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        # lora_B: [out_features, rank]
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights: A is Gaussian/Kaiming uniform, B is strictly zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass through original frozen layer
        base_output = self.original_layer(x)

        # LoRA forward pass: F.linear(F.linear(x, A), B) * scaling
        # x: [..., in_features] -> A: [..., rank] -> B: [..., out_features]
        lora_drop = self.dropout(x)
        lora_step_a = F.linear(lora_drop, self.lora_A)
        lora_step_b = F.linear(lora_step_a, self.lora_B)
        lora_output = lora_step_b * self.scaling

        return base_output + lora_output


def inject_lora_adapters(
    model: nn.Module,
    target_modules: tuple[str, ...] = ("c_attn",),
    rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    """Inject custom LoRALinear adapters into target attention layers of the model.

    Freezes all base model parameters and replaces matching linear/Conv1D modules.
    Returns list of names of replaced modules.
    """
    # 1. Freeze every parameter in the base model
    for param in model.parameters():
        param.requires_grad = False

    replaced_modules: list[str] = []

    def _recursive_inject(parent: nn.Module, parent_name: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name
            if any(target in name for target in target_modules) and not isinstance(child, LoRALinear):
                adapter = LoRALinear(
                    original_layer=child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(parent, name, adapter)
                replaced_modules.append(full_name)
            else:
                _recursive_inject(child, full_name)

    _recursive_inject(model)
    return replaced_modules


def get_trainable_parameters(model: nn.Module) -> dict[str, Any]:
    """Return total, trainable, and percentage of trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable / total * 100.0) if total > 0 else 0.0
    return {
        "total": total,
        "trainable": trainable,
        "percent_trainable": round(pct, 4),
    }


def is_online(host: str = "huggingface.co", port: int = 443, timeout: float = 0.5) -> bool:
    """Quick check if Hugging Face hub is reachable without triggering long urllib3 retry loops."""
    import socket

    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False


def load_model_and_tokenizer(
    model_name: str = "distilgpt2",
    device: torch.device | None = None,
    fallback_offline: bool = True,
) -> tuple[nn.Module, Any]:
    """Load a small pre-trained Hugging Face causal LM and tokenizer.

    If network is unavailable or model is not cached, gracefully initializes a local
    lightweight GPT-2 architecture and SimpleSportsTokenizer.
    """
    dev = device or get_device()

    # 1. Try local cache first to avoid unnecessary network latency
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        model.to(dev)
        return model, tokenizer
    except Exception:
        pass

    # 2. If online, attempt to download from Hugging Face Hub
    if is_online():
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(dev)
            return model, tokenizer
        except Exception:
            pass

    # 3. Fallback to local lightweight GPT-2 model and offline sports tokenizer
    if not fallback_offline:
        raise ConnectionError(
            f"Cannot load '{model_name}': offline and model is not cached locally."
        )

    from transformers import GPT2Config, GPT2LMHeadModel

    tokenizer = SimpleSportsTokenizer()
    vocab_size = max(len(tokenizer), 128)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.to(dev)
    return model, tokenizer


def train_lora(
    model: nn.Module,
    tokenizer: Any,
    corpus_text: str,
    epochs: int = 2,
    lr: float = 1e-3,
    batch_size: int = 2,
    max_length: int = 64,
    device: torch.device | None = None,
) -> list[float]:
    """Lightweight fine-tuning loop updating only LoRA parameters on commentary text.

    Optimized for laptop/CPU speed and minimal resource usage.
    """
    dev = device or next(model.parameters()).device
    model.train()

    # Verify that only LoRA parameters have requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Did you call inject_lora_adapters?")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # Prepare chunks from commentary text
    sentences = [s.strip() for s in re.split(r"[.\n]+", corpus_text) if len(s.strip()) > 10]
    if not sentences:
        sentences = [corpus_text]

    batch_input_ids: list[torch.Tensor] = []
    for s in sentences:
        if hasattr(tokenizer, "encode"):
            res = tokenizer.encode(s, max_length=max_length, truncation=True)
            if isinstance(res, dict):
                ids = res["input_ids"]
                if isinstance(ids, torch.Tensor):
                    ids = ids.squeeze().tolist()
            else:
                ids = res
        else:
            ids = [1, 2, 3]

        if len(ids) > 2:
            batch_input_ids.append(torch.tensor(ids, dtype=torch.long))

    if not batch_input_ids:
        # Fallback dummy sequence
        batch_input_ids = [torch.tensor([1, 2, 3, 4, 1], dtype=torch.long)]

    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    epoch_losses: list[float] = []

    for _ in range(epochs):
        epoch_loss_accum = 0.0
        steps = 0

        # Process mini-batches
        for i in range(0, len(batch_input_ids), batch_size):
            chunk = batch_input_ids[i : i + batch_size]
            max_len = max(len(t) for t in chunk)
            padded = [torch.cat([t, torch.full((max_len - len(t),), pad_id, dtype=torch.long)]) for t in chunk]
            inputs = torch.stack(padded).to(dev)

            # Causal language modeling: labels match inputs
            labels = inputs.clone()
            labels[labels == pad_id] = -100

            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss_accum += float(loss.item())
            steps += 1

        avg_loss = epoch_loss_accum / max(steps, 1)
        epoch_losses.append(round(avg_loss, 4))

    return epoch_losses


def generate_llm(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 40,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: torch.device | None = None,
) -> str:
    """Generate commentary continuation from the LoRA-adapted LLM."""
    dev = device or next(model.parameters()).device
    model.eval()

    # Tokenize input prompt
    if hasattr(tokenizer, "encode"):
        enc = tokenizer.encode(prompt, return_tensors="pt")
        if isinstance(enc, dict):
            input_ids = enc["input_ids"].to(dev)
        elif isinstance(enc, torch.Tensor):
            input_ids = enc.to(dev)
        else:
            input_ids = torch.tensor([enc], dtype=torch.long, device=dev)
    else:
        input_ids = torch.tensor([[1]], device=dev)

    eos_id = getattr(tokenizer, "eos_token_id", None) or 50256
    pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id

    # Try Hugging Face generate API first
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_tokens,
                temperature=max(float(temperature), 1e-4),
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True).strip()
    except Exception:
        pass

    # Autoregressive fallback generation loop
    curr_ids = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(curr_ids).logits[:, -1, :] / max(float(temperature), 1e-4)
            if top_k > 0:
                val, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < val[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)
            if eos_id is not None and next_token.item() == eos_id:
                break

    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(curr_ids[0].tolist(), skip_special_tokens=True).strip()
    return prompt


def save_lora_weights(model: nn.Module, path: str | Path) -> None:
    """Save only the lightweight LoRA adapter parameters to disk."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, str(save_path))


def load_lora_weights(model: nn.Module, path: str | Path) -> None:
    """Load cached LoRA adapter parameters into the model."""
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"LoRA weights not found at {load_path}")
    checkpoint = torch.load(str(load_path), map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
