from clinspacy import parse


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
