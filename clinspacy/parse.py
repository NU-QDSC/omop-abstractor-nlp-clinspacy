import re
from spacy.language import Language
from typing import List, Dict, Tuple
from textabstractor.dataclasses import (
    AbstractionSchemaMetaData,
    AbstractionSchema,
    AbstractorSection,
    Variant,
)


def parse_schema(
    schema: AbstractionSchema, meta_schema: AbstractionSchemaMetaData, nlp: Language
) -> Tuple[Dict, List[Dict]]:
    rule_type = meta_schema.abstractor_rule_type
    object_type = meta_schema.abstractor_object_type

    name_patterns = {}
    value_patterns = []

    if rule_type == "name/value":
        name_patterns = parse_name_list_schema(schema, nlp)
    if object_type in ["list", "radio button list"]:
        value_patterns = parse_value_list_schema(schema, nlp)
    elif object_type == "number":
        value_patterns = get_number_schema(schema)
    elif object_type == "date":
        value_patterns = get_date_schema(schema)

    return name_patterns, value_patterns


def get_number_schema(schema: AbstractionSchema) -> List[Dict]:
    return [
        {
            "predicate": schema.predicate,
            "patterns": [
                [{"LIKE_NUM": True}, {"ORTH": "%", "OP": "?"}],
            ],
            "value": "number",
            "rule_type": "value",
            "object_type": "number",
        }
    ]


def get_date_schema(schema: AbstractionSchema) -> List[Dict]:
    return [
        {
            "predicate": schema.predicate,
            "patterns": [[{"TEXT": {"REGEX": r"^\d{1,2}/\d{1,2}/\d\d(\d\d)?$"}}]],
            "value": "date",
            "rule_type": "value",
            "object_type": "date",
        }
    ]


def parse_variant(variant: Variant, nlp: Language) -> List[Dict]:
    pattern = []
    doc = nlp(variant.value)
    for token in doc:
        p = {}
        if variant.case_sensitive is True:
            p["ORTH"] = token.text
        elif token.is_punct:
            p["ORTH"] = token.orth_
            p["OP"] = "?"
        else:
            p["LEMMA"] = token.lemma_
        pattern.append(p)
    return pattern


def parse_name_list_schema(schema: AbstractionSchema, nlp: Language) -> Dict:
    name_patterns = {
        "predicate": schema.predicate,
        "patterns": [],
        "value": schema.preferred_name,
        "rule_type": "name",
        "object_type": "list",
    }
    with nlp.select_pipes(enable=["tagger", "attribute_ruler", "lemmatizer"]):
        # this is to deal with values like "glioblastoma (9448/3)"
        value = re.sub(r"\(.+\)", "", schema.preferred_name).strip()
        pattern = parse_variant(Variant(value=value, case_sensitive=False), nlp)
        name_patterns["patterns"].append(pattern)
        for variant in schema.predicate_variants:
            pattern = parse_variant(variant, nlp)
            name_patterns["patterns"].append(pattern)
    return name_patterns


def parse_value_list_schema(schema: AbstractionSchema, nlp: Language) -> List[Dict]:
    value_patterns = []
    with nlp.select_pipes(enable=["tagger", "attribute_ruler", "lemmatizer"]):
        for object_value in schema.object_values:
            object_patterns = {
                "predicate": schema.predicate,
                "patterns": [],
                "value": object_value.value,
                "rule_type": "value",
                "object_type": "list",
            }
            # this is to deal with values like "glioblastoma (9448/3)"
            value = re.sub(r"\(.+\)", "", object_value.value).strip()
            pattern = parse_variant(
                Variant(value=value, case_sensitive=object_value.case_sensitive), nlp
            )
            object_patterns["patterns"].append(pattern)
            for variant in object_value.object_value_variants:
                pattern = parse_variant(variant, nlp)
                object_patterns["patterns"].append(pattern)
            value_patterns.append(object_patterns)
    return value_patterns


def parse_section(
    section_metadata: AbstractorSection, nlp: Language
) -> Tuple[str, List[List[Dict]]]:
    patterns = []
    if section_metadata.section_mention_type == "Alphabetic":
        patterns.append(
            {"REGEX_TEXT": re.compile(r"^.{0,21}\b[A-Z][.:)]", re.MULTILINE)}
        )
    elif section_metadata.section_mention_type == "Token":
        with nlp.select_pipes(enable=["tagger", "attribute_ruler", "lemmatizer"]):
            for variant in section_metadata.section_name_variants:
                doc = nlp(variant.name)
                pattern = []
                for token in doc:
                    if token.idx == 0:
                        p = {"ORTH": token.text, "IS_SENT_START": True}
                    else:
                        p = {"ORTH": token.text}
                    pattern.append(p)
                patterns.append(pattern)
    return section_metadata.name, patterns
