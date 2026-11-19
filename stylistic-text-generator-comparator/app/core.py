"""Core implementation of N-gram model, milestone orchestration, and comparison logic."""
from __future__ import annotations

import collections
import random
from pathlib import Path
from typing import Any

from app.llm import (
    LoRALinear,
    generate_llm,
    get_trainable_parameters,
    inject_lora_adapters,
    load_model_and_tokenizer,
    train_lora,
)
from app.storage import clean_text, load_corpus, tokenize_llm, tokenize_ngram


# --- N-Gram Model from Scratch ---

class NGramModel:
    """A flexible N-gram language model built from scratch using standard Python collections.

    Supports:
    - Arbitrary n-gram orders (bigram, trigram, 4-gram, etc.)
    - Recursive Katz-style backoff for unseen contexts
    - Optional Laplace (add-k) smoothing
    - Temperature-scaled probabilistic token sampling
    """

    def __init__(self, n: int = 3, laplace_smoothing: float = 0.0) -> None:
        if n < 1:
            raise ValueError(f"N-gram order must be >= 1, got {n}")
        self.n = n
        self.laplace_smoothing = float(laplace_smoothing)
        # Primary counts: context tuple (length n-1) -> Counter of next tokens
        self.counts: dict[tuple[str, ...], collections.Counter[str]] = collections.defaultdict(collections.Counter)
        # Multi-order counts for backoff: order -> context -> Counter
        self.order_counts: dict[int, dict[tuple[str, ...], collections.Counter[str]]] = {
            k: collections.defaultdict(collections.Counter) for k in range(1, self.n + 1)
        }
        self.vocab: set[str] = set()
        self.unigram_counts: collections.Counter[str] = collections.Counter()
        self.total_tokens: int = 0

    def train(self, corpus: str | list[str]) -> None:
        """Train the N-gram model on the provided sports commentary corpus.

        Can accept either a raw text string or a list of pre-tokenized words.
        """
        if isinstance(corpus, str):
            tokens = tokenize_ngram(corpus, lowercase=True)
        else:
            tokens = [tok.lower() if tok.isalnum() else tok for tok in corpus]

        if not tokens:
            return

        self.total_tokens += len(tokens)
        for tok in tokens:
            self.vocab.add(tok)
            self.unigram_counts[tok] += 1

        # Build counts for order n and all lower orders (for robust backoff)
        for k in range(1, self.n + 1):
            for i in range(len(tokens) - k + 1):
                ngram = tokens[i : i + k]
                context = tuple(ngram[:-1])
                target = ngram[-1]
                self.order_counts[k][context][target] += 1
                if k == self.n:
                    self.counts[context][target] += 1

    def get_distribution(self, context: tuple[str, ...]) -> dict[str, float]:
        """Compute the next-token probability distribution given a context tuple.

        Applies backoff to shorter contexts if the full context is unseen.
        """
        # Ensure context matches length n-1
        ctx = tuple(context[-(self.n - 1) :]) if self.n > 1 else ()

        # Backoff search from order n-1 down to 0
        counter: collections.Counter[str] | None = None
        for k in range(len(ctx), -1, -1):
            sub_ctx = ctx[len(ctx) - k :] if k > 0 else ()
            order_k = k + 1
            if sub_ctx in self.order_counts.get(order_k, {}):
                counter = self.order_counts[order_k][sub_ctx]
                if counter:
                    break

        if not counter:
            counter = self.unigram_counts

        total = sum(counter.values())
        if total == 0:
            if not self.vocab:
                return {}
            uniform = 1.0 / len(self.vocab)
            return {w: uniform for w in self.vocab}

        if self.laplace_smoothing > 0:
            k = self.laplace_smoothing
            denom = total + k * len(self.vocab)
            dist: dict[str, float] = {}
            for w in self.vocab:
                c = counter.get(w, 0)
                dist[w] = (c + k) / denom
            return dist

        return {w: count / total for w, count in counter.items()}

    def predict_next(
        self,
        context: tuple[str, ...],
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> str:
        """Sample the next token given context and sampling temperature."""
        if seed is not None:
            random.seed(seed)

        dist = self.get_distribution(context)
        if not dist:
            return "."

        words = list(dist.keys())
        probs = list(dist.values())

        temp = max(float(temperature), 1e-4)
        if abs(temp - 1.0) > 1e-3:
            # Temperature scaling: p_i^(1/T) / sum(p_j^(1/T))
            exp_probs = [p ** (1.0 / temp) for p in probs]
            sum_exp = sum(exp_probs)
            probs = [p / sum_exp for p in exp_probs] if sum_exp > 0 else probs

        return random.choices(words, weights=probs, k=1)[0]

    def generate(
        self,
        prompt: str,
        max_tokens: int = 40,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> str:
        """Generate text starting from the prompt up to max_tokens."""
        if seed is not None:
            random.seed(seed)

        prompt_tokens = tokenize_ngram(prompt, lowercase=True) if prompt.strip() else []
        generated = list(prompt_tokens)

        for _ in range(max_tokens):
            if self.n > 1:
                ctx = tuple(generated[-(self.n - 1) :]) if len(generated) >= self.n - 1 else tuple(generated)
            else:
                ctx = ()

            next_token = self.predict_next(ctx, temperature=temperature)
            generated.append(next_token)
            # Stop if multiple end-of-sentence markers occur
            if len(generated) > len(prompt_tokens) + 10 and next_token in (".", "!", "?") and random.random() < 0.3:
                break

        # Detokenize tokens into readable commentary text
        text = ""
        for i, tok in enumerate(generated):
            if not text or tok in ".,!?;:":
                text += tok
            elif tok.startswith("'"):
                text += tok
            else:
                text += " " + tok

        # Capitalize first letter of sentences
        sentences = [s.strip().capitalize() for s in text.split(". ") if s.strip()]
        return ". ".join(sentences)


# --- Milestone Implementations ---

def milestone_1(corpus_path: str | None = None) -> dict[str, Any]:
    """Milestone 1: Prepare and preprocess the sports commentary text corpus."""
    raw_text = load_corpus(corpus_path)
    cleaned = clean_text(raw_text)
    ngram_tokens = tokenize_ngram(cleaned, lowercase=True)
    llm_tokens = tokenize_llm(cleaned)

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "ngram_tokens": ngram_tokens,
        "ngram_token_count": len(ngram_tokens),
        "llm_tokens": llm_tokens,
    }


def milestone_2(
    corpus_path: str | None = None,
    prompt: str = "Here comes",
    max_tokens: int = 25,
    n: int = 3,
) -> dict[str, Any]:
    """Milestone 2: Train the custom N-gram model from scratch and generate commentary text."""
    corpus = load_corpus(corpus_path)
    model = NGramModel(n=n)
    model.train(corpus)
    generated = model.generate(prompt=prompt, max_tokens=max_tokens)

    return {
        "model": model,
        "prompt": prompt,
        "generated_text": generated,
        "ngram_order": n,
        "vocab_size": len(model.vocab),
    }


def milestone_3(
    model_name: str = "distilgpt2",
    rank: int = 4,
    alpha: float = 16.0,
) -> dict[str, Any]:
    """Milestone 3: Load pre-trained LLM and inject custom LoRALinear adapters from scratch."""
    model, tokenizer = load_model_and_tokenizer(model_name)
    injected = inject_lora_adapters(model, target_modules=("c_attn",), rank=rank, alpha=alpha)
    stats = get_trainable_parameters(model)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "injected_modules": injected,
        "parameter_stats": stats,
    }


def milestone_4(
    model: Any = None,
    tokenizer: Any = None,
    corpus_path: str | None = None,
    epochs: int = 2,
    rank: int = 4,
) -> dict[str, Any]:
    """Milestone 4: Fine-tune only the LoRA adapter weights on the sports commentary corpus."""
    if model is None or tokenizer is None:
        m3_res = milestone_3(rank=rank)
        model, tokenizer = m3_res["model"], m3_res["tokenizer"]

    corpus = load_corpus(corpus_path)
    losses = train_lora(model, tokenizer, corpus_text=corpus, epochs=epochs)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "losses": losses,
        "epochs": epochs,
    }


def milestone_5(
    model: Any = None,
    tokenizer: Any = None,
    prompt: str = "Rashford bursts down the wing",
    max_tokens: int = 25,
) -> dict[str, Any]:
    """Milestone 5: Generate sports commentary text from the LoRA-adapted LLM."""
    if model is None or tokenizer is None:
        m4_res = milestone_4(epochs=1)
        model, tokenizer = m4_res["model"], m4_res["tokenizer"]

    generated = generate_llm(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    return {
        "prompt": prompt,
        "generated_text": generated,
    }


def milestone_6(
    prompt: str = "What an unbelievable goal",
    corpus_path: str | None = None,
    max_tokens: int = 30,
) -> dict[str, Any]:
    """Milestone 6: Qualitative comparison interface between N-gram and LoRA-adapted LLM."""
    # 1. Train and generate from N-gram model
    m2_res = milestone_2(corpus_path=corpus_path, prompt=prompt, max_tokens=max_tokens, n=3)
    ngram_output = m2_res["generated_text"]

    # 2. Train and generate from LoRA-adapted LLM
    m4_res = milestone_4(corpus_path=corpus_path, epochs=1)
    m5_res = milestone_5(model=m4_res["model"], tokenizer=m4_res["tokenizer"], prompt=prompt, max_tokens=max_tokens)
    lora_output = m5_res["generated_text"]

    comparison = {
        "prompt": prompt,
        "ngram_output": ngram_output,
        "lora_output": lora_output,
        "ngram_vocab_size": m2_res["vocab_size"],
        "lora_losses": m4_res["losses"],
        "summary": "N-gram captures local phrasing patterns; LoRA LLM generalizes semantic flow and style.",
    }
    return comparison


def format_side_by_side(
    prompt: str,
    ngram_text: str,
    lora_text: str,
    ngram_stats: dict[str, Any] | None = None,
    lora_stats: dict[str, Any] | None = None,
    width: int = 45,
) -> str:
    """Format outputs side-by-side in ASCII box format for sports journalists in the terminal."""
    import textwrap

    ngram_lines = textwrap.wrap(ngram_text, width=width) or ["(No output)"]
    lora_lines = textwrap.wrap(lora_text, width=width) or ["(No output)"]

    max_lines = max(len(ngram_lines), len(lora_lines))
    ngram_lines += [""] * (max_lines - len(ngram_lines))
    lora_lines += [""] * (max_lines - len(lora_lines))

    sep = "=" * (width * 2 + 5)
    header = f"{'N-GRAM MODEL OUTPUT':<{width}} | {'LoRA-ADAPTED LLM OUTPUT':<{width}}"
    sub_sep = "-" * width + "-+-" + "-" * width

    stat_ngram = ""
    if ngram_stats:
        stat_ngram = f"Order: {ngram_stats.get('order', 3)} | Vocab: {ngram_stats.get('vocab_size', 0)}"

    stat_lora = ""
    if lora_stats:
        stat_lora = f"Rank: {lora_stats.get('rank', 4)} | Trainable%: {lora_stats.get('percent_trainable', '0%')}"

    stat_line = f"{stat_ngram:<{width}} | {stat_lora:<{width}}"

    rows = []
    for nl, ll in zip(ngram_lines, lora_lines):
        rows.append(f"{nl:<{width}} | {ll:<{width}}")

    out = [
        sep,
        f"PROMPT: \"{prompt}\"",
        sep,
        header,
        stat_line,
        sub_sep,
        *rows,
        sep,
    ]
    return "\n".join(out)
