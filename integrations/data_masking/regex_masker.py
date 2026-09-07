# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic (regex-based) PII masking layer.

Covers PII with a fixed, well-defined shape — the kind a language model
would be overkill (and non-deterministic) for. Free-form entities that don't
follow a fixed pattern (person names, company names, addresses) are handled
by `llm_ner_masker.py` instead; see `pipeline.py` for how the two combine.

This layer requires no model, no GPU, and no network access, so it is safe
to run as a first pass even before the closed-network LLM stage is set up.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    category: str


# Order matters: more specific patterns first so a later, looser pattern
# cannot swallow part of an already-matched span (spans are applied
# leftmost-longest, see `find_spans`).
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("RRN", re.compile(r"\b\d{6}[-\s]?[1-4]\d{6}\b")),  # 주민등록번호
    ("BRN", re.compile(r"\b\d{3}-\d{2}-\d{5}\b")),  # 사업자등록번호
    ("CARD", re.compile(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b")),  # 신용카드번호
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"\b01[016789][-\s]?\d{3,4}[-\s]?\d{4}\b")),  # 휴대폰
    ("PHONE", re.compile(r"\b0\d{1,2}[-\s]\d{3,4}[-\s]\d{4}\b")),  # 일반전화
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def find_spans(text: str) -> list[Span]:
    """Find all pattern matches, resolving overlaps by earliest-start then longest-match."""
    candidates = []
    for category, pattern in PATTERNS:
        for match in pattern.finditer(text):
            candidates.append(Span(match.start(), match.end(), match.group(), category))

    candidates.sort(key=lambda s: (s.start, -(s.end - s.start)))

    spans: list[Span] = []
    last_end = -1
    for span in candidates:
        if span.start >= last_end:
            spans.append(span)
            last_end = span.end
    return spans


def mask(text: str) -> tuple[str, list[Span]]:
    """Replace every matched span with `[<CATEGORY>]` and return (masked_text, spans)."""
    spans = find_spans(text)
    out = []
    cursor = 0
    for span in spans:
        out.append(text[cursor : span.start])
        out.append(f"[{span.category}]")
        cursor = span.end
    out.append(text[cursor:])
    return "".join(out), spans
