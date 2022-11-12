import json, os
from pathlib import Path
from rich import print
from typing import Dict
from clinspacy.extract import *  # noqa: F401
from importlib_resources import files
from clinspacy import abstract
import textabstractor
import textabstractor_testdata.prostate as prostate_data
from textabstractor.dataclasses import (
    AbstractionSchema,
    SuggestRequest,
)


dir_path = Path(os.path.dirname(os.path.realpath(__file__)))


def load_prostate_schemas() -> Dict[int, AbstractionSchema]:
    schemas = {}
    for i in range(2055, 2069):
        json_text = files(prostate_data).joinpath(f"{str(i)}.json").read_text()
        json_dict = json.loads(json_text)
        schema = AbstractionSchema(**json_dict["abstractor_abstraction_schema"])
        schemas[i] = schema
    return schemas


def test_relation_extractor(suggest_request, abstractor):
    name_patterns = {
        "predicate": "has_tumor_size",
        "patterns": [
            [{"LEMMA": "tumor"}, {"LEMMA": "extent"}],
            [{"LEMMA": "tumor"}, {"LEMMA": "size"}],
        ],
        "value": "tumor size",
        "rule_type": "name",
        "object_type": "list",
    }
    value_patterns = {
        "predicate": "has_tumor_size",
        "patterns": [
            [{"LIKE_NUM": True}, {"ORTH": "%", "OP": "?"}],
            [{"lower": "large"}],
        ],
        "value": "number",
        "rule_type": "value",
        "object_type": "number",
    }
    name_patterns["value_patterns"] = [value_patterns]
    abstractor.span_ruler.add(
        f"{name_patterns['rule_type']}:{name_patterns['value']}", name_patterns
    )
    doc = abstractor.nlp(
        """
    The patient has DCIS. Tumor size: large, 3.5cm, less then 4.0cm.
    Certainly less than 5.0cm. Tumor extent is 10%, very small.
    """
    )

    spans = doc.spans["name:tumor size"]

    assert len(spans) == 2
    assert "value_map" in spans.attrs
    assert len(spans.attrs["value_map"]) == 2
    assert len(spans.attrs["value_map"][spans[0]]) == 3
    assert spans.attrs["value_map"][spans[0]][0].label_ == "large"
    assert spans.attrs["value_map"][spans[0]][1].label_ == "3.5"
    assert spans.attrs["value_map"][spans[0]][2].label_ == "4.0"
    assert len(spans.attrs["value_map"][spans[1]]) == 1
    assert spans.attrs["value_map"][spans[1]][0].label_ == "10%"


def test_no_xxx_identified(abstractor):
    schemas = load_prostate_schemas()
    json_dict = json.loads(files(prostate_data).joinpath("request.json").read_text())
    request = SuggestRequest(**json_dict)
    request.text = """
    No lymphovascular invasion identified.
    """

    for schema_meta_data in request.abstractor_abstraction_schemas:
        textabstractor.textabstract.schema_cache[
            schema_meta_data.abstractor_abstraction_schema_uri
        ] = (
            schema_meta_data,
            schemas[schema_meta_data.abstractor_abstraction_schema_id],
        )

    response = abstract.process_text(request)
    assert len(response.suggestions) == 3
    assert (
        len(
            [
                r
                for r in response.suggestions
                if r.predicate == "has_lymphovascular_invasion"
            ]
        )
        == 3
    )
    assert (
        len(
            [
                r
                for r in response.suggestions
                if r.type == "name" and r.assertion == "absent"
            ]
        )
        == 1
    )
    assert (
        len(
            [
                r
                for r in response.suggestions
                if r.type == "value" and r.assertion == "present"
            ]
        )
        == 2
    )


def test_not_identified():
    schemas = load_prostate_schemas()
    request = SuggestRequest(
        **json.loads((dir_path / "data/prostate/request.json").read_text())
    )
    request.text = """
    A. Prostate and seminal vesicles, radical prostatectomy:
    ? Extraprostatic Extension (In Fat): Not Identified"
    ? Extraprostatic Extension (In Fat): PRESENT"
    """

    for schema_meta_data in request.abstractor_abstraction_schemas:
        textabstractor.textabstract.schema_cache[
            schema_meta_data.abstractor_abstraction_schema_uri
        ] = (
            schema_meta_data,
            schemas[schema_meta_data.abstractor_abstraction_schema_id],
        )

    response = abstract.process_text(request)
    assert len(response.suggestions) == 4
    assert (
        len(
            [
                r
                for r in response.suggestions
                if r.predicate == "has_extraprostatic_extension"
            ]
        )
        == 4
    )
    assert len([r for r in response.suggestions if r.type == "name"]) == 2
    assert len([r for r in response.suggestions if r.type == "value"]) == 2
    assert len([r for r in response.suggestions if r.value == "not identified"]) == 1
    assert len([r for r in response.suggestions if r.value == "present"]) == 1
