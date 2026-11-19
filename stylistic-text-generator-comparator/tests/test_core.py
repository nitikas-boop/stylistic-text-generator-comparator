"""Unit tests for the Stylistic Text Generator Comparator milestones."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from app.core import (
    NGramModel,
    format_side_by_side,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    milestone_6,
)
from app.llm import (
    LoRALinear,
    SimpleSportsTokenizer,
    generate_llm,
    get_trainable_parameters,
    inject_lora_adapters,
    load_lora_weights,
    save_lora_weights,
    train_lora,
)
from app.storage import clean_text, load_corpus, tokenize_llm, tokenize_ngram


# --- Milestone 1: Prepare a small corpus of sports commentary text ---

def test_milestone_1():
    """Verify corpus loading, cleaning, and tokenization for both models."""
    res = milestone_1()
    assert "raw_text" in res
    assert "cleaned_text" in res
    assert "ngram_tokens" in res
    assert res["ngram_token_count"] > 10

    # Test clean_text helper
    dirty_text = "Here  comes   Manchester!   Rashford’s shot “curls” into   the net.\n\n\nGoal!"
    cleaned = clean_text(dirty_text)
    assert "  " not in cleaned
    assert "Rashford's" in cleaned
    assert '"curls"' in cleaned

    # Test N-gram tokenization preserves words and sentence punctuation
    tokens = tokenize_ngram(cleaned, lowercase=True)
    assert "here" in tokens
    assert "!" in tokens
    assert "goal" in tokens

    # Test LLM tokenization output
    tok_res = tokenize_llm("Rashford bursts forward.")
    assert "input_ids" in tok_res
    assert "attention_mask" in tok_res


# --- Milestone 2: Implement the N-gram model and generate text from it ---

def test_milestone_2():
    """Verify N-gram model training, counts, probability distributions, and text generation."""
    sample_text = (
        "Rashford bursts down the flank. Rashford bursts into the box. "
        "Rashford crosses the ball to Fernandes."
    )
    model = NGramModel(n=3)
    model.train(sample_text)

    # Check vocabulary and n-gram counts
    assert "rashford" in model.vocab
    assert "bursts" in model.vocab
    context = ("rashford", "bursts")
    assert context in model.counts
    assert model.counts[context]["down"] == 1
    assert model.counts[context]["into"] == 1

    # Check probability distribution sums to 1.0
    dist = model.get_distribution(context)
    assert abs(sum(dist.values()) - 1.0) < 1e-4
    assert dist["down"] == 0.5
    assert dist["into"] == 0.5

    # Check text generation builds on prompt
    generated = model.generate(prompt="Rashford bursts", max_tokens=15, temperature=0.7, seed=42)
    assert isinstance(generated, str)
    assert len(generated.split()) >= 2
    assert "rashford" in generated.lower()

    # Check milestone_2 runner
    m2_res = milestone_2(prompt="Here comes", max_tokens=10, n=2)
    assert "generated_text" in m2_res
    assert m2_res["vocab_size"] > 0


# --- Milestone 3: Load pre-trained LLM and implement LoRA adapter injection ---

def test_milestone_3():
    """Verify LoRALinear class, parameter shapes, freezing, and adapter injection."""
    # 1. Test LoRALinear isolated on a standard linear layer
    in_features, out_features, rank, alpha = 16, 32, 4, 16.0
    base_linear = nn.Linear(in_features, out_features)
    lora_linear = LoRALinear(base_linear, rank=rank, alpha=alpha)

    # Base layer must be frozen
    for p in base_linear.parameters():
        assert p.requires_grad is False

    # LoRA parameters must be trainable and have correct dimensions
    assert lora_linear.lora_A.shape == (rank, in_features)
    assert lora_linear.lora_B.shape == (out_features, rank)
    assert lora_linear.lora_A.requires_grad is True
    assert lora_linear.lora_B.requires_grad is True
    assert lora_linear.scaling == alpha / rank

    # Zero initialization of B ensures initial forward pass matches base layer exactly
    x = torch.randn(2, in_features)
    with torch.no_grad():
        base_out = base_linear(x)
        lora_out = lora_linear(x)
        assert torch.allclose(base_out, lora_out, atol=1e-5)

    # 2. Test injection into a miniature GPT-2 model
    config = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config)
    replaced = inject_lora_adapters(model, target_modules=("c_attn",), rank=rank, alpha=alpha)
    assert len(replaced) == 2  # 2 layers with c_attn

    stats = get_trainable_parameters(model)
    assert stats["trainable"] > 0
    assert stats["percent_trainable"] < 100.0

    # 3. Test milestone_3 helper
    m3_res = milestone_3(rank=4)
    assert len(m3_res["injected_modules"]) > 0
    assert m3_res["parameter_stats"]["trainable"] > 0


# --- Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus ---

def test_milestone_4(tmp_path):
    """Verify fine-tuning loop trains only LoRA parameters while base parameters stay frozen."""
    tokenizer = SimpleSportsTokenizer()
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    inject_lora_adapters(model, target_modules=("c_attn",), rank=2)

    # Snapshot base weight and LoRA weight before training
    first_lora_module = None
    for m in model.modules():
        if isinstance(m, LoRALinear):
            first_lora_module = m
            break
    assert first_lora_module is not None

    base_weight_before = first_lora_module.original_layer.weight.clone()
    lora_b_before = first_lora_module.lora_B.clone()

    corpus_snippet = "Rashford bursts down the wing. What an unbelievable goal in stoppage time!"
    losses = train_lora(model, tokenizer, corpus_snippet, epochs=2, lr=1e-2, batch_size=1)
    assert len(losses) == 2
    assert all(isinstance(l, float) for l in losses)

    # Base weights must remain strictly untouched
    assert torch.equal(first_lora_module.original_layer.weight, base_weight_before)
    # LoRA B weights must have received gradient updates and shifted away from zero
    assert not torch.equal(first_lora_module.lora_B, lora_b_before)

    # Test weight saving and loading
    weight_file = tmp_path / "lora_test.pt"
    save_lora_weights(model, weight_file)
    assert weight_file.exists()

    # Re-inject fresh adapter and load weights
    model_fresh = GPT2LMHeadModel(config)
    inject_lora_adapters(model_fresh, target_modules=("c_attn",), rank=2)
    load_lora_weights(model_fresh, weight_file)


# --- Milestone 5: Generate text from the LoRA-adapted LLM ---

def test_milestone_5():
    """Verify text generation from the fine-tuned LoRA model."""
    tokenizer = SimpleSportsTokenizer()
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    inject_lora_adapters(model, target_modules=("c_attn",), rank=2)

    prompt = "Rashford bursts down the wing"
    output = generate_llm(model, tokenizer, prompt=prompt, max_tokens=15, temperature=0.7)
    assert isinstance(output, str)
    assert len(output) > 0


# --- Milestone 6: Provide qualitative comparison interface for generated texts ---

def test_milestone_6():
    """Verify qualitative comparison structure and side-by-side terminal formatter."""
    comparison = milestone_6(prompt="Goal in the final minute", max_tokens=15)
    assert "prompt" in comparison
    assert "ngram_output" in comparison
    assert "lora_output" in comparison
    assert "summary" in comparison

    # Verify side-by-side table formatter output
    table = format_side_by_side(
        prompt="Counter-attack opportunity",
        ngram_text="Rashford bursts down the left flank at speed.",
        lora_text="He skips past the defender and strikes it into the net.",
        ngram_stats={"order": 3, "vocab_size": 150},
        lora_stats={"rank": 4, "percent_trainable": "0.12%"},
    )
    assert "N-GRAM MODEL OUTPUT" in table
    assert "LoRA-ADAPTED LLM OUTPUT" in table
    assert "Counter-attack opportunity" in table
