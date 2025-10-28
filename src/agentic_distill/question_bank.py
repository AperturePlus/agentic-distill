"""Utilities for loading and sampling pre-generated case/question banks."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


@dataclass(frozen=True)
class QuestionBankEntry:
    """Structured representation of a single question/case."""

    uid: str
    payload: Dict[str, Any]

    @property
    def fingerprint(self) -> str:
        issue = self.payload.get("issue", "")
        tier = self.payload.get("customer_tier", "")
        key = f"{issue}|{tier}".strip().lower()
        return key or self.uid


class QuestionBank:
    """In-memory question bank that avoids reusing entries within a run."""

    def __init__(self, path: Path | str, *, seed: Optional[int] = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Question bank not found at {self.path}")

        self._entries = self._load_entries(self.path)
        if not self._entries:
            raise ValueError(f"Question bank at {self.path} is empty.")

        self._random = random.Random(seed)
        self._available_indices = list(range(len(self._entries)))
        self._used_fingerprints: Set[str] = set()

    @staticmethod
    def _load_entries(path: Path) -> List[QuestionBankEntry]:
        entries: List[QuestionBankEntry] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = str(data.get("id") or data.get("uid") or data.get("issue") or len(entries))
                entries.append(QuestionBankEntry(uid=uid, payload=data))
        return entries

    def __len__(self) -> int:
        return len(self._entries)

    def remaining(self) -> int:
        return len(self._available_indices)

    def sample(self) -> Dict[str, Any]:
        """Return a unique entry; reshuffle when exhausted."""

        if not self._available_indices:
            # Reset the pool but keep track of fingerprints to minimise duplicates
            self._available_indices = list(range(len(self._entries)))

        retries = len(self._available_indices)
        while retries > 0 and self._available_indices:
            index = self._random.choice(self._available_indices)
            self._available_indices.remove(index)
            entry = self._entries[index]
            if entry.fingerprint in self._used_fingerprints:
                retries -= 1
                continue
            self._used_fingerprints.add(entry.fingerprint)
            return entry.payload

        # If all fingerprints are exhausted, fall back to a random entry
        fallback = self._random.choice(self._entries)
        return fallback.payload

    def preview(self, count: int = 3) -> List[Dict[str, Any]]:
        """Return a small list of entries for inspection."""

        indices = self._random.sample(range(len(self._entries)), k=min(count, len(self._entries)))
        return [self._entries[idx].payload for idx in indices]

