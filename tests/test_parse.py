import re
from clinspacy import parse

# The regex used in parse.py to strip ICD-O codes from values
ICDO_CODE_PATTERN = r"\([Cc]?\d[\d./?]+\)"


class TestIcdoCodeStripping:
    """Test that the regex strips ICD-O codes but preserves descriptive parenthesized text."""

    def test_strips_morphology_code(self):
        value = "glioblastoma (9448/3)"
        assert re.sub(ICDO_CODE_PATTERN, "", value).strip() == "glioblastoma"

    def test_strips_morphology_code_with_behavior(self):
        value = "infiltrating duct carcinoma (8500/3)"
        assert re.sub(ICDO_CODE_PATTERN, "", value).strip() == "infiltrating duct carcinoma"

    def test_strips_topography_code_lowercase(self):
        value = "breast (c50.9)"
        assert re.sub(ICDO_CODE_PATTERN, "", value).strip() == "breast"

    def test_strips_topography_code_uppercase(self):
        value = "ampulla of vater (C24.1)"
        assert re.sub(ICDO_CODE_PATTERN, "", value).strip() == "ampulla of vater"

    def test_strips_metastatic_code(self):
        value = "adenocarcinoma, metastatic (8140/6)"
        assert re.sub(ICDO_CODE_PATTERN, "", value).strip() == "adenocarcinoma, metastatic"

    def test_preserves_exocrine(self):
        value = "Pancreatic (Exocrine) Cancer Staging Summary"
        assert re.sub(ICDO_CODE_PATTERN, "", value).strip() == value

    def test_preserves_dcis(self):
        value = "_pTis (DCIS)"
        assert re.sub(ICDO_CODE_PATTERN, "", value).strip() == value

    def test_preserves_paget(self):
        value = "_pTis (Paget)"
        assert re.sub(ICDO_CODE_PATTERN, "", value).strip() == value

    def test_preserves_staging_iplus(self):
        value = "N0 (i+)"
        assert re.sub(ICDO_CODE_PATTERN, "", value).strip() == value

    def test_preserves_staging_molplus(self):
        value = "N0 (mol+)"
        assert re.sub(ICDO_CODE_PATTERN, "", value).strip() == value

    def test_preserves_single_letter(self):
        value = "nodes (n)"
        assert re.sub(ICDO_CODE_PATTERN, "", value).strip() == value

    def test_strips_short_topography_code(self):
        value = "gum (c03)"
        assert re.sub(ICDO_CODE_PATTERN, "", value).strip() == "gum"


def test_parse_value_list_schema(schemas, suggest_request, abstractor):
    meta_schema = [
        s
        for s in suggest_request.abstractor_abstraction_schemas
        if s.abstractor_abstraction_schema_id == 285
    ][0]
    schema = schemas[285]
    name_patterns, value_patterns = parse.parse_schema(
        schema, meta_schema, abstractor.nlp
    )
    assert len(name_patterns) == 0
    assert len(value_patterns) == 22
    assert len([p for vp in value_patterns for p in vp["patterns"]]) == 149


def test_parse_name_value_list_schema(abstractor, suggest_request, schemas):
    meta_schema = [
        s
        for s in suggest_request.abstractor_abstraction_schemas
        if s.abstractor_abstraction_schema_id == 291
    ][0]
    schema = schemas[291]
    name_patterns, value_patterns = parse.parse_schema(
        schema, meta_schema, abstractor.nlp
    )
    assert len(name_patterns["patterns"]) == 4
    assert len(value_patterns) == 321
    assert len([p for vp in value_patterns for p in vp["patterns"]]) == 1579


def test_parse_name_value_number_schema(abstractor, suggest_request, schemas):
    meta_schema = [
        s
        for s in suggest_request.abstractor_abstraction_schemas
        if s.abstractor_abstraction_schema_id == 292
    ][0]
    schema = schemas[292]
    name_patterns, value_patterns = parse.parse_schema(
        schema, meta_schema, abstractor.nlp
    )
    assert len(name_patterns["patterns"]) == 2
    assert len(value_patterns) == 1


def test_parse_name_value_date_schema(abstractor, suggest_request, schemas):
    meta_schema = [
        s
        for s in suggest_request.abstractor_abstraction_schemas
        if s.abstractor_abstraction_schema_id == 299
    ][0]
    schema = schemas[299]
    name_patterns, value_patterns = parse.parse_schema(
        schema, meta_schema, abstractor.nlp
    )
    assert len(name_patterns["patterns"]) == 3
    assert len(value_patterns) == 1


def test_abstract_with_value_list_schema(abstractor, suggest_request, schemas):
    meta_schema = [
        s
        for s in suggest_request.abstractor_abstraction_schemas
        if s.abstractor_abstraction_schema_id == 285
    ][0]
    schema = schemas[285]
    name_patterns, value_patterns = parse.parse_schema(
        schema, meta_schema, abstractor.nlp
    )
    for pattern in value_patterns:
        abstractor.span_ruler.add(f"{pattern['value']}", pattern)
    doc = abstractor.nlp(
        """
    The patient has colloid adenocarcinoma and lobular carcinoma.
    She also has infiltrating duct carcinoma and mucinous adenocarcinoma.
    """
    )
    assert len(doc.ents) == 0
    values = [v[i] for _, v in doc.spans.items() for i in range(len(v)) if len(v) > 0]
    assert len(values) == 4
