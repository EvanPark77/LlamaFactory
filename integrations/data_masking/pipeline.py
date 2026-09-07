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

"""Two-layer masking pipeline: regex first, then LLM-detected free-form entities.

`detect_entities` is injected rather than imported directly, so this module
has no hard dependency on `transformers`/`torch` (and can be unit-tested with
a fake detector — see `tests/test_pipeline.py`). In production, pass
`GemmaLocalNER(...).detect_entities` from `llm_ner_masker.py`.

Audit logging follows the same principle used elsewhere in this project's
ECR compliance work: never store the original PII text, only a hash of it,
so the audit trail cannot itself become a PII leak.
"""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from regex_masker import mask as regex_mask


@dataclass
class AuditEntry:
    category: str
    tag: str
    span_hash: str
    detector: str  # "regex" or "llm_ner"


@dataclass
class MaskResult:
    masked_text: str
    audit_log: list[AuditEntry] = field(default_factory=list)


def _hash_span(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def mask_document(
    text: str,
    detect_entities: Callable[[str], list[dict]] | None = None,
) -> MaskResult:
    """Mask PII in `text`.

    1. Regex layer masks fixed-shape PII (phone numbers, IDs, emails, ...).
    2. If `detect_entities` is given, it runs on the regex-masked text (so it
       never has to see already-redacted numbers) and returns free-form
       entities (PERSON/ORG/LOCATION); every literal occurrence of each
       entity string is replaced with a stable per-document tag, so the same
       company name maps to the same tag everywhere in the document.

    Passing `detect_entities=None` runs the regex layer only — useful in a
    closed-network environment where the LLM stage isn't staged yet, or for
    a fast dev-time check.
    """
    audit_log: list[AuditEntry] = []

    masked_text, regex_spans = regex_mask(text)
    for span in regex_spans:
        audit_log.append(
            AuditEntry(
                category=span.category, tag=f"[{span.category}]", span_hash=_hash_span(span.text), detector="regex"
            )
        )

    if detect_entities is None:
        return MaskResult(masked_text=masked_text, audit_log=audit_log)

    entities = detect_entities(masked_text)

    # Longest text first, so "가상기업A 대표이사" doesn't get partially consumed
    # before "가상기업A" is tagged.
    seen_texts = sorted(
        {e["text"] for e in entities if e["text"] and not e["text"].startswith("[")}, key=len, reverse=True
    )
    entity_type_by_text = {e["text"]: e["type"] for e in entities}

    tag_by_text: dict[str, str] = {}
    counters: dict[str, int] = {}

    for entity_text in seen_texts:
        if entity_text not in masked_text:
            continue  # model hallucinated a span not actually present
        entity_type = entity_type_by_text[entity_text]
        counters[entity_type] = counters.get(entity_type, 0) + 1
        tag = f"[{entity_type}_{counters[entity_type]}]"
        tag_by_text[entity_text] = tag
        audit_log.append(
            AuditEntry(category=entity_type, tag=tag, span_hash=_hash_span(entity_text), detector="llm_ner")
        )

    if tag_by_text:
        pattern = re.compile("|".join(re.escape(t) for t in tag_by_text))
        masked_text = pattern.sub(lambda m: tag_by_text[m.group()], masked_text)

    return MaskResult(masked_text=masked_text, audit_log=audit_log)
