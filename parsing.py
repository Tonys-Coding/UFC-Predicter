"""Validated primitive parsers shared by HTML and archive ingestion."""

from __future__ import annotations

import re


def count_pair(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s+of\s+(\d+)\s*", str(text))
    if not match or int(match[1]) > int(match[2]):
        raise ValueError("Invalid landed/attempted statistic.")
    return int(match[1]), int(match[2])


def duration_seconds(round_number: int, clock: str, time_format: str) -> int:
    match = re.fullmatch(r"(\d{1,2}):([0-5]\d)", clock)
    schedule = re.search(r"\(([\d-]+)\)", time_format)
    if not match or not schedule:
        raise ValueError("Unsupported historical round format.")
    lengths = [int(value) * 60 for value in schedule[1].split("-")]
    elapsed = int(match[1]) * 60 + int(match[2])
    if round_number < 1 or round_number > len(lengths) or elapsed > lengths[round_number - 1]:
        raise ValueError("Finish time exceeds scheduled round duration.")
    total = sum(lengths[: round_number - 1]) + elapsed
    if total <= 0:
        raise ValueError("Fight has no recorded duration.")
    return total
