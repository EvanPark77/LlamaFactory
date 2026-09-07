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

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from regex_masker import mask  # noqa: E402


def test_masks_business_registration_number():
    masked, spans = mask("사업자등록번호: 123-45-67890 입니다.")
    assert masked == "사업자등록번호: [BRN] 입니다."
    assert [s.category for s in spans] == ["BRN"]


def test_masks_resident_registration_number():
    masked, _ = mask("담당자 주민번호는 900101-1234567 입니다.")
    assert masked == "담당자 주민번호는 [RRN] 입니다."


def test_masks_phone_email_and_ip():
    text = "문의: 010-1234-5678, contact@example.com, 접속 IP 192.168.0.1"
    masked, spans = mask(text)
    assert masked == "문의: [PHONE], [EMAIL], 접속 IP [IP]"
    assert {s.category for s in spans} == {"PHONE", "EMAIL", "IP"}


def test_masks_credit_card_number():
    masked, _ = mask("카드번호 1234-5678-9012-3456 결제 완료")
    assert masked == "카드번호 [CARD] 결제 완료"


def test_no_false_positive_on_plain_indicator_values():
    text = "부채비율 = 250, 유동비율 = 80"
    masked, spans = mask(text)
    assert masked == text
    assert spans == []


def test_overlap_resolution_prefers_earliest_then_longest():
    # a credit-card-shaped number should not also get partially re-matched by
    # a shorter pattern starting inside it
    masked, spans = mask("번호 1234-5678-9012-3456 끝")
    assert len(spans) == 1
    assert spans[0].category == "CARD"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK  {name}")
    print("all tests passed")
