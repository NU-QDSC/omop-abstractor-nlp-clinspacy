import spacy
import pytest
from spacy.lang.en import English
from clinspacy.match import *


@pytest.fixture(scope="session")
def patterns() -> Dict:
    return {
        "id": "id1",
        "predicate": "predicate1",
        "patterns": [
            [{"LOWER": "hello"}],
            [{"LOWER": "hello"}, {"LOWER": "world"}],
        ],
    }


def test_span_matcher_on_doc(patterns):
    span_matcher = SpanMatcher(patterns["predicate"], patterns)
    assert span_matcher.name == patterns["predicate"]
    assert span_matcher.patterns == patterns
    nlp = English()
    doc = nlp("This is a test. Hello world!")
    group = span_matcher.match(doc)
    assert len(group) == 2
    assert group.attrs["id"] == "id1"
    assert group.attrs["predicate"] == "predicate1"
    texts = [s.text for s in group]
    assert "Hello world" in texts
    assert "Hello" in texts

    group = span_matcher.match(doc, keep_longest=True)
    assert len(group) == 1
    assert group.attrs["id"] == "id1"
    assert group.attrs["predicate"] == "predicate1"
    assert group[0].text == "Hello world"


def test_span_matcher_on_span(patterns):
    span_matcher = SpanMatcher(patterns["predicate"], patterns)
    assert span_matcher.name == patterns["predicate"]
    assert span_matcher.patterns == patterns
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("This is a test. Hello world!")
    for sent in doc.sents:
        span_doc = Span(doc, sent.start, sent.end).as_doc()
        group = span_matcher.match(span_doc)
        for span in group:
            assert doc[span.start + sent.start : span.end + sent.start]


def test_span_ruler(patterns):
    nlp = English()
    span_ruler = nlp.add_pipe("span_match_ruler")
    span_ruler.add("entities1", patterns)
    span_ruler.add("entities2", patterns)
    assert "span_match_ruler" in nlp.pipe_names
    doc = nlp("This is a test. Hello world!")
    assert len(doc.spans) == 2
