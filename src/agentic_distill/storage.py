"""Dataset storage utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .types import Episode


class DatasetWriter:
    """Writes episodes to disk according to the configured format."""

    def __init__(self, *, base_dir: Path, format: str = "jsonl", shard_size: int = 500):
        self.base_dir = Path(base_dir)
        self.format = format
        self.shard_size = shard_size
        self._buffer: list[Episode] = []
        self._shard_index = 0
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, episode: Episode) -> None:
        """Buffer an episode and flush if the shard is full."""

        self._buffer.append(episode)
        if len(self._buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        """Persist buffered episodes to disk."""

        if not self._buffer:
            return

        shard_path = self.base_dir / f"shard-{self._shard_index:05d}.{self.format}"
        serializable = [episode.to_serializable() for episode in self._buffer]
        if self.format == "jsonl":
            with shard_path.open("w", encoding="utf-8") as fp:
                for item in serializable:
                    fp.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif self.format == "parquet":
            df = pd.DataFrame(serializable)
            df.to_parquet(shard_path, index=False)
        else:
            raise ValueError(f"Unsupported dataset format '{self.format}'.")

        self._buffer.clear()
        self._shard_index += 1

    def finalize(self) -> None:
        """Flush remaining episodes and close resources."""

        self.flush()

    def write_many(self, episodes: Iterable[Episode]) -> None:
        """Convenience method for streaming multiple episodes."""

        for episode in episodes:
            self.write(episode)
        self.finalize()

