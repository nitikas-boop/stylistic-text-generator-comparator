"""Data storage and sports commentary corpus loading, cleaning, and tokenization."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_CORPUS_TEXT = """Here comes Manchester on the counter-attack! Rashford bursts down the left flank at blistering speed.
He skips past the defender with a quick stepover and drives towards the byline.
He looks up and delivers a tantalizing cross into the dangerous penalty box.
Fernandes rises highest above the center-backs and headers it towards goal!
What an unbelievable fingertip save by the goalkeeper! That is pure world-class agility.
The rebound falls loose to the edge of the eighteen-yard box.
A thunderous first-time strike from distance! It curls with venom into the top right corner of the net!
Goal! What an absolute screamer in stoppage time!
The crowd erupts in sheer pandemonium! Absolute scenes here at the stadium as the referee checks his watch.
The visiting defense was caught completely flat-footed on the break.
He controlled the tempo, picked out the runner, and smashed it home with surgical precision.
The final whistle blows and what a heroic performance we have witnessed tonight!

Curry brings the basketball across half court with ice in his veins.
He calls for the high screen at the top of the key and shakes his defender with a wicked crossover.
Steps back behind the three-point arc with hands in his face and launches it high into the air.
Bang! Pure water through the nylon! What a clutch shot from downtown!
The arena explodes as the timeout is immediately called by the opposing coach.
On the next possession, transition opportunity on the fast break.
A no-look behind-the-back pass to the cutting power forward.
He takes off from outside the paint and finishes with a monstrous two-handed slam dunk!
That brought the entire bench to their feet! Defensive stop, fast transition, and ruthless execution.
With five seconds left on the game clock, the pressure is palpable.
The inbound pass is deflected! He dives across the hardwood to secure the loose ball.
Game, set, match! An unforgettable night of basketball drama!"""


class Store:
    """A tiny key-value store in one JSON file (no server, easy to read). Values can be anything JSON can hold."""

    def __init__(self, path: str | None = None):
        self.path = Path(path or os.environ.get("DATA_PATH", "data/app.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def put(self, key: str, value: Any) -> None:
        data = self._read()
        data[key] = value
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self._read().get(key, default)

    def all(self) -> dict:
        return dict(sorted(self._read().items()))


def clean_text(text: str) -> str:
    """Clean and normalize sports commentary text.

    - Replaces typographic quotes/apostrophes with standard ASCII characters.
    - Normalizes irregular whitespace, tabs, and duplicate newlines.
    - Strips leading and trailing whitespace while preserving sentence punctuation.
    """
    if not text:
        return ""

    # Normalize unicode apostrophes and quotation marks
    cleaned = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # Normalize multiple whitespace characters within lines
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    # Normalize excessive empty lines to double newlines (paragraphs)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def load_corpus(file_path: str | Path | None = None) -> str:
    """Load and clean sports commentary text from a local file or default path.

    If file_path is None or does not exist, loads the bundled sports commentary corpus.
    """
    if file_path is not None:
        path = Path(file_path)
        if path.exists() and path.is_file():
            content = path.read_text(encoding="utf-8", errors="replace")
            return clean_text(content)

    # Check for default data directory
    default_path = Path(__file__).resolve().parent.parent / "data" / "sports_commentary.txt"
    if default_path.exists():
        content = default_path.read_text(encoding="utf-8", errors="replace")
        return clean_text(content)

    return clean_text(DEFAULT_CORPUS_TEXT)


def save_corpus(text: str, file_path: str | Path) -> None:
    """Save cleaned sports commentary text to a file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(clean_text(text), encoding="utf-8")


def tokenize_ngram(text: str, lowercase: bool = True) -> list[str]:
    """Tokenize text into words and punctuation tokens suitable for N-gram models.

    Words (including contractions like "don't" or "world-class") and sentence punctuation
    (!, ?, ., ,, ;, :) are captured as distinct tokens.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # Regex captures word tokens (with optional internal hyphen or apostrophe) or punctuation
    tokens = re.findall(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|[.,!?;:]", cleaned)
    if lowercase:
        tokens = [token.lower() if token.isalnum() else token for token in tokens]
    return tokens


def tokenize_llm(
    text: str,
    tokenizer=None,
    max_length: int = 128,
    return_tensors: str = "pt",
) -> Any:
    """Tokenize commentary text for transformer-based LLMs.

    If a tokenizer is provided, uses its standard encoding API.
    If tokenizer is None, uses a fallback tokenization scheme returning dictionary
    with 'input_ids' and 'attention_mask'.
    """
    cleaned = clean_text(text)
    if tokenizer is not None:
        return tokenizer(
            cleaned,
            truncation=True,
            max_length=max_length,
            padding="max_length" if max_length else False,
            return_tensors=return_tensors,
        )

    # Fallback when tokenizer is not passed: simple whitespace token mapping
    words = tokenize_ngram(cleaned, lowercase=False)
    # Simple hash-based mapping for offline/mock usage
    input_ids = [min((hash(w) % 1000) + 1, 999) for w in words[:max_length]]
    if not input_ids:
        input_ids = [0]
    import torch

    tensor_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.ones_like(tensor_ids)
    if return_tensors == "pt":
        return {"input_ids": tensor_ids, "attention_mask": attention_mask}
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
