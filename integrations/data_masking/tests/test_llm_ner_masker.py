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

"""Tests for the pure parsing/prompt logic — no torch/transformers required.

`GemmaLocalNER` itself needs a real local checkpoint and a GPU, so it is not
covered here; these tests cover the model-agnostic boundary (prompt in,
JSON out) so the combination logic in `pipeline.py` can be trusted without
needing to run Gemma-4.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_ner_masker import build_messages, parse_ner_response  # noqa: E402


def test_parses_clean_json_array():
    raw = '[{"text": "가상기업A", "type": "ORG"}, {"text": "홍길동", "type": "PERSON"}]'
    entities = parse_ner_response(raw)
    assert entities == [
        {"text": "가상기업A", "type": "ORG"},
        {"text": "홍길동", "type": "PERSON"},
    ]


def test_tolerates_markdown_code_fence_and_commentary():
    raw = '물론입니다. 다음과 같이 찾았습니다:\n```json\n[{"text": "서울특별시", "type": "LOCATION"}]\n```\n'
    entities = parse_ner_response(raw)
    assert entities == [{"text": "서울특별시", "type": "LOCATION"}]


def test_empty_array_when_nothing_found():
    assert parse_ner_response("[]") == []


def test_drops_entries_with_unknown_type():
    raw = '[{"text": "부채비율", "type": "INDICATOR"}, {"text": "홍길동", "type": "PERSON"}]'
    entities = parse_ner_response(raw)
    assert entities == [{"text": "홍길동", "type": "PERSON"}]


def test_returns_empty_on_malformed_json():
    assert parse_ner_response("모델이 JSON을 만들지 못했습니다.") == []


def test_build_messages_shape():
    messages = build_messages("가상기업A의 담당자는 홍길동입니다.")
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "가상기업A의 담당자는 홍길동입니다."}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK  {name}")
    print("all tests passed")
