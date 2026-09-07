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

"""End-to-end pipeline tests using a fake LLM detector (no GPU/model needed).

The fake stands in for `GemmaLocalNER.detect_entities` — same input/output
contract (text in, list of {"text", "type"} dicts out) — so these tests
exercise the exact combination logic that runs in production, just without
a real Gemma-4 call.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import mask_document  # noqa: E402


def fake_detector(_text: str) -> list[dict]:
    return [
        {"text": "가상기업A", "type": "ORG"},
        {"text": "홍길동", "type": "PERSON"},
    ]


def test_regex_only_when_no_detector_given():
    text = "가상기업A 담당자 홍길동, 연락처 010-1234-5678"
    result = mask_document(text, detect_entities=None)
    assert result.masked_text == "가상기업A 담당자 홍길동, 연락처 [PHONE]"
    assert len(result.audit_log) == 1
    assert result.audit_log[0].detector == "regex"


def test_combines_regex_and_llm_layers():
    text = "가상기업A 담당자 홍길동, 연락처 010-1234-5678, 사업자번호 123-45-67890"
    result = mask_document(text, detect_entities=fake_detector)

    assert "가상기업A" not in result.masked_text
    assert "홍길동" not in result.masked_text
    assert "010-1234-5678" not in result.masked_text
    assert "123-45-67890" not in result.masked_text
    assert "[ORG_1]" in result.masked_text
    assert "[PERSON_1]" in result.masked_text
    assert "[PHONE]" in result.masked_text
    assert "[BRN]" in result.masked_text

    detectors_used = {entry.detector for entry in result.audit_log}
    assert detectors_used == {"regex", "llm_ner"}


def test_same_entity_gets_same_tag_every_occurrence():
    text = "가상기업A는 오늘 실적을 발표했다. 가상기업A의 주가는 상승했다."
    result = mask_document(text, detect_entities=fake_detector)
    assert result.masked_text.count("[ORG_1]") == 2
    assert "가상기업A" not in result.masked_text


def test_audit_log_never_contains_raw_pii():
    text = "가상기업A 담당자 홍길동, 사업자번호 123-45-67890"
    result = mask_document(text, detect_entities=fake_detector)
    for entry in result.audit_log:
        assert "가상기업A" not in entry.span_hash
        assert "홍길동" not in entry.span_hash
        assert "123-45-67890" not in entry.span_hash
        assert len(entry.span_hash) == 16  # hash, not the original text


def test_ignores_hallucinated_entity_not_present_in_text():
    def hallucinating_detector(_text: str) -> list[dict]:
        return [{"text": "존재하지않는회사", "type": "ORG"}]

    text = "가상기업A 관련 문서"
    result = mask_document(text, detect_entities=hallucinating_detector)
    assert result.masked_text == text
    assert result.audit_log == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK  {name}")
    print("all tests passed")
