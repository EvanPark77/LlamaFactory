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

"""Free-form entity detection (PERSON/ORG/LOCATION) via a locally-hosted Gemma-4.

`regex_masker.py` only catches PII with a fixed shape (phone numbers, IDs,
emails). Company names, person names, and addresses do not follow a fixed
pattern, so this module asks a locally-hosted instruction-tuned LLM to find
them instead.

Closed-network (폐쇄망) safety: this module NEVER makes a network call. It
only loads a model that already exists on local disk (`local_files_only`
below), and sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` at import time so a
transformers call cannot silently fall back to the Hugging Face Hub even if
a caller forgets to pass the right flags. See `README.md` for how to stage
the Gemma-4 checkpoint into a closed network before this can run there.

`transformers`/`torch` are imported lazily inside `GemmaLocalNER` so that
`parse_ner_response` (pure string/JSON logic) can be unit-tested on a machine
that has neither installed.
"""

import json
import os
import re


# Defense in depth: force offline mode even if the caller forgets to. Real
# network isolation must still be enforced at the network layer (폐쇄망) —
# this only prevents accidental calls from this process.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ENTITY_CATEGORIES = ("PERSON", "ORG", "LOCATION")

NER_SYSTEM_PROMPT = (
    "다음 텍스트에서 사람 이름(PERSON), 회사/기관명(ORG), 지명/주소(LOCATION)를 "
    "모두 찾아라. 각 항목은 텍스트에 실제로 등장하는 표현 그대로 추출한다. "
    "다른 설명 없이 JSON 배열만 출력한다. 형식: "
    '[{"text": "발견된 표현", "type": "PERSON|ORG|LOCATION"}]. '
    "찾은 것이 없으면 빈 배열 []을 출력한다."
)

_JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


def build_messages(text: str) -> list[dict]:
    """Build chat-format messages for the `gemma4` template.

    Same `system` + `user` shape as the conventions used elsewhere in this
    repo's `data/dataset_info.json`.
    """
    return [
        {"role": "system", "content": NER_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def parse_ner_response(raw_text: str) -> list[dict]:
    """Parse the model's response into a validated list of {"text", "type"} dicts.

    Tolerates markdown code fences and leading/trailing commentary around the
    JSON array, since instruction-tuned models don't always follow the
    "no other text" instruction exactly.
    """
    match = _JSON_ARRAY_PATTERN.search(raw_text)
    if not match:
        return []

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    entities = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        entity_type = item.get("type")
        if isinstance(text, str) and text and entity_type in ENTITY_CATEGORIES:
            entities.append({"text": text, "type": entity_type})
    return entities


class GemmaLocalNER:
    """Loads Gemma-4 from a local path and runs it as an entity detector.

    `model_path` must be a local directory (already staged into the closed
    network — see README.md), never a Hugging Face Hub repo id: passing a
    hub id here would only work by accident (a leftover cache), and it does
    not belong in a 폐쇄망 codepath even conceptually.
    """

    def __init__(self, model_path: str, device: str = "cuda", max_new_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if model_path.count("/") == 1 and not os.path.isdir(model_path):
            raise ValueError(
                f"'{model_path}' looks like a Hugging Face Hub id, not a local path. "
                "Point this at a directory staged on local disk (see README.md's "
                "closed-network staging steps) — this class never downloads a model."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.device = device
        self.max_new_tokens = max_new_tokens

    def detect_entities(self, text: str) -> list[dict]:
        prompt = self.tokenizer.apply_chat_template(build_messages(text), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return parse_ner_response(generated)
