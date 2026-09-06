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

"""Rule-based KG builder for the credit-evaluation ontology demo.

Parses the fixed key: value document format under `sample_docs/` and populates
an RDF graph (ontology schema + instances) conforming to
`ontology/credit_evaluation.ttl`.

This is a deterministic, regex-based extractor meant to prove the pipeline
end-to-end with synthetic documents. For real documents with free-form prose,
replace `parse_indicator_doc` / `parse_case_doc` with an LLM-based extractor
(KAG / SAC-KG style: schema-guided extraction + verification against the
ontology), keeping the same output graph shape.
"""

import argparse
import re
from pathlib import Path

from rdflib import RDF, RDFS, Graph, Literal, Namespace
from rdflib.namespace import XSD


CRED = Namespace("http://saltlux.example.org/ontology/credit-eval#")

RISK_DIRECTION_MAP = {
    "높을수록위험": "high_is_risk",
    "낮을수록위험": "low_is_risk",
}

FIELD_LINE_PATTERN = re.compile(r"^(?P<key>[^:\s][^:]*):\s*(?P<value>.*)$")
OBS_LINE_PATTERN = re.compile(r"^\s*(?P<indicator>\S+)\s*=\s*(?P<value>[\d.]+)\s*$")


def parse_indicator_doc(text: str) -> dict:
    """Parse one indicator document.

    Each `key: value` line is one field; a `정의:` field may continue on the
    following indented lines until the next `key:` line.
    """
    fields: dict[str, str] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        match = FIELD_LINE_PATTERN.match(raw_line)
        if match and not raw_line.startswith((" ", "\t")):
            current_key = match.group("key").strip()
            fields[current_key] = match.group("value").strip()
        elif current_key is not None and raw_line.strip():
            fields[current_key] += " " + raw_line.strip()

    required = ["지표명", "변수ID", "정의", "위험방향", "위험임계값", "심각임계값"]
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"indicator document missing fields {missing}:\n{text[:200]}")

    return {
        "name": fields["지표명"],
        "var_id": fields["변수ID"],
        "definition": fields["정의"],
        "direction": RISK_DIRECTION_MAP[fields["위험방향"]],
        "threshold_warning": float(fields["위험임계값"]),
        "threshold_severe": float(fields["심각임계값"]),
    }


def parse_case_doc(text: str) -> list[dict]:
    """Line-based parser for one or more `기업명:` / `관측일:` / `관측:` blocks."""
    cases: list[dict] = []
    current: dict | None = None
    in_observations = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("기업명:"):
            if current is not None:
                cases.append(current)
            current = {"company": line.split(":", 1)[1].strip(), "date": None, "observations": []}
            in_observations = False
        elif line.startswith("관측일:") and current is not None:
            current["date"] = line.split(":", 1)[1].strip()
        elif line.startswith("관측:"):
            in_observations = True
        elif in_observations and current is not None:
            obs_match = OBS_LINE_PATTERN.match(raw_line)
            if obs_match:
                current["observations"].append(
                    {"indicator_name": obs_match.group("indicator"), "value": float(obs_match.group("value"))}
                )
            elif line:
                in_observations = False

    if current is not None:
        cases.append(current)
    return cases


def build_graph(sample_docs_dir: Path, ontology_path: Path) -> Graph:
    graph = Graph()
    graph.parse(ontology_path, format="turtle")
    graph.bind("cred", CRED)

    indicator_by_name: dict[str, dict] = {}

    for path in sorted(sample_docs_dir.glob("indicator_*.txt")):
        indicator = parse_indicator_doc(path.read_text(encoding="utf-8"))
        indicator_by_name[indicator["name"]] = indicator

        indicator_uri = CRED[f"IndicatorType_{indicator['var_id']}"]
        graph.add((indicator_uri, RDF.type, CRED.IndicatorType))
        graph.add((indicator_uri, RDFS.label, Literal(indicator["name"], lang="ko")))
        graph.add((indicator_uri, CRED.definition, Literal(indicator["definition"])))
        graph.add((indicator_uri, CRED.riskDirection, Literal(indicator["direction"])))
        graph.add(
            (indicator_uri, CRED.thresholdWarning, Literal(indicator["threshold_warning"], datatype=XSD.decimal))
        )
        graph.add((indicator_uri, CRED.thresholdSevere, Literal(indicator["threshold_severe"], datatype=XSD.decimal)))

    obs_counter = 0
    for path in sorted(sample_docs_dir.glob("case_*.txt")):
        for case in parse_case_doc(path.read_text(encoding="utf-8")):
            company_uri = CRED[f"Company_{case['company']}"]
            graph.add((company_uri, RDF.type, CRED.Company))
            graph.add((company_uri, RDFS.label, Literal(case["company"], lang="ko")))

            for obs in case["observations"]:
                indicator = indicator_by_name.get(obs["indicator_name"])
                if indicator is None:
                    raise ValueError(
                        f"case document references unknown indicator '{obs['indicator_name']}' "
                        "— add a matching indicator_*.txt document first"
                    )

                obs_counter += 1
                obs_uri = CRED[f"Observation_{obs_counter}"]
                indicator_uri = CRED[f"IndicatorType_{indicator['var_id']}"]

                graph.add((obs_uri, RDF.type, CRED.IndicatorObservation))
                graph.add((obs_uri, CRED.ofIndicatorType, indicator_uri))
                graph.add((obs_uri, CRED.hasValue, Literal(obs["value"], datatype=XSD.decimal)))
                graph.add((obs_uri, CRED.variableId, Literal(indicator["var_id"])))
                graph.add((obs_uri, CRED.observedAt, Literal(case["date"], datatype=XSD.date)))
                graph.add((company_uri, CRED.hasObservation, obs_uri))

    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-docs-dir", type=Path, default=Path(__file__).parent / "sample_docs")
    parser.add_argument(
        "--ontology", type=Path, default=Path(__file__).parent.parent / "ontology" / "credit_evaluation.ttl"
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "kg_instance.ttl")
    args = parser.parse_args()

    graph = build_graph(args.sample_docs_dir, args.ontology)
    graph.serialize(destination=args.output, format="turtle")
    print(f"wrote {len(graph)} triples to {args.output}")


if __name__ == "__main__":
    main()
