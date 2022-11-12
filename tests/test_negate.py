import spacy
import pytest
from spacy.lang.en import English
from clinspacy import parse
from clinspacy.negate import *


@pytest.fixture
def pseudo_negations():
    return config["negation"]["pseudo_negations"]


@pytest.fixture
def pre_negations():
    return config["negation"]["pre_negations"]


@pytest.fixture
def post_negations():
    return config["negation"]["post_negations"]


@pytest.fixture
def terminators():
    return config["negation"]["terminators"]


@pytest.fixture
def sample_text():
    return """
    We found DCIS, but intraductal carcinoma and adenoid cystic carcinoma were ruled out.
    There was no significant change to DCIS, although intraductal carcinoma was found.
    DCIS was not found.
    No DCIS was found.
    """


def test_smoke(pseudo_negations, pre_negations, post_negations, terminators):
    nlp = spacy.load("en_core_web_sm")

    assert len(pseudo_negations) == 24
    assert len(pre_negations) == 81
    assert len(post_negations) == 11
    assert len(terminators) == 34

    negex = Negex(
        nlp,
        "negex",
        pseudo_negations=pseudo_negations,
        pre_negations=pre_negations,
        post_negations=post_negations,
        terminators=terminators,
    )
    assert negex.pseudo_neg_matcher
    assert negex.pre_neg_matcher
    assert negex.post_neg_matcher
    assert negex.term_matcher


@pytest.mark.parametrize(
    "patterns, num_patterns, num_matches",
    [
        ("pseudo_negations", 24, 1),
        ("pre_negations", 81, 4),
        ("post_negations", 11, 2),
        ("terminators", 34, 2),
    ],
)
def test_parse_phrases(
    abstractor, sample_text, patterns, num_patterns, num_matches, request
):
    doc = abstractor.nlp(sample_text)
    pattern_map = Negex.parse_phrases(
        patterns, request.getfixturevalue(patterns), abstractor.nlp
    )
    assert pattern_map["name"] == patterns
    assert len(pattern_map["patterns"]) == num_patterns
    span_matcher = SpanMatcher(pattern_map["name"], pattern_map)
    matches = span_matcher.match(doc)
    assert len(matches) == num_matches


def test_find_negations(
    sample_text, pseudo_negations, pre_negations, post_negations, terminators
):
    nlp = English()
    nlp.add_pipe("pysbd")
    nlp.add_pipe("span_match_ruler")
    doc = nlp(sample_text)
    negex = Negex(
        nlp,
        "negex",
        pseudo_negations=pseudo_negations,
        pre_negations=pre_negations,
        post_negations=post_negations,
        terminators=terminators,
    )
    neg_matches = negex.find_negations(doc)
    expected = {
        "pseudo_negations": ["no significant change"],
        "pre_negations": ["No"],
        "post_negations": ["were ruled out", "was not"],
        "terminators": ["but", "although"],
    }
    assert [str(p) for p in neg_matches["pseudo_negations"]] == expected[
        "pseudo_negations"
    ]
    assert [str(p) for p in neg_matches["pre_negations"]] == expected["pre_negations"]
    assert [str(p) for p in neg_matches["post_negations"]] == expected["post_negations"]
    assert [str(p) for p in neg_matches["terminators"]] == expected["terminators"]


def test_negation(abstractor, sample_text, suggest_request, schemas):
    abstractor.nlp.remove_pipe("negex")
    negex = abstractor.nlp.add_pipe("negex")
    assert Span.has_extension("negated")

    meta_schema = [
        s
        for s in suggest_request.abstractor_abstraction_schemas
        if s.abstractor_abstraction_schema_id == 285
    ][0]
    schema = schemas[285]
    name_patterns, value_patterns = parse.parse_schema(
        schema, meta_schema, abstractor.nlp
    )
    for vp in value_patterns:
        abstractor.span_ruler.add(vp["value"], vp)

    doc = abstractor.nlp(sample_text)
    span_groups = [
        group
        for _, group in doc.spans.items()
        if group and group.attrs.get("rule_type", "") == "value"
    ]
    spans = [s for g in span_groups for s in g]
    spans.sort()
    assert spans[0]._.negated is False
    assert spans[1]._.negated is True
    assert spans[2]._.negated is True
    assert spans[3]._.negated is False
    assert spans[4]._.negated is False
    assert spans[5]._.negated is True
    assert spans[6]._.negated is True
