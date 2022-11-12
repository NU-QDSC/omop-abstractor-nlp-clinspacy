from typing import List, Dict
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Span, SpanGroup
from spacy.tokens.doc import Doc


# ----------------------------------------------------------------------------------------------------------------------
def get_covered_spans(spans: SpanGroup, cover_span: Span):
    return [
        span
        for span in spans
        if span.start >= cover_span.start and span.end <= cover_span.end
    ]


# ----------------------------------------------------------------------------------------------------------------------
def covers(span1: Span, span2: Span, strict: bool = True) -> bool:
    """
    Does span1 cover span2?
    :param span1:
    :param span2:
    :param strict:
    :return:
    """
    if (
        span1.start <= span2.start
        and span1.end >= span2.end
        and (not strict or len(span1) > len(span2))
    ):
        return True
    return False


# ----------------------------------------------------------------------------------------------------------------------
def filter_covered(possible_covers: SpanGroup, spans: SpanGroup, strict: bool = True) -> SpanGroup:
    remove_indices = []
    for idx, span in enumerate(spans):
        for s in possible_covers:
            if covers(s, span, strict):
                remove_indices.append(idx)

    uncovered_spans = []
    for idx, span in enumerate(spans):
        if idx not in remove_indices:
            uncovered_spans.append(span)

    return SpanGroup(spans.doc, spans=uncovered_spans)


# ----------------------------------------------------------------------------------------------------------------------
class SpanMatcher:
    def __init__(self, name: str, patterns: Dict):
        self._name = name
        self._patterns = patterns

    @property
    def name(self):
        return self._name

    @property
    def patterns(self):
        return self._patterns

    @staticmethod
    def keep_longest(group: SpanGroup) -> SpanGroup:
        if not group.has_overlap:
            return group
        longest_spans = []
        for span in group:
            covers = [
                s
                for s in group
                if s.start <= span.start and s.end >= span.end and len(s) > len(span)
            ]
            if len(covers) == 0:
                longest_spans.append(span)
        return SpanGroup(group.doc, spans=longest_spans, attrs=group.attrs)

    def match(self, doc: Doc, keep_longest: bool = False) -> SpanGroup:
        matcher = Matcher(doc.vocab)
        matcher.add(self.name, self.patterns["patterns"])
        matches = matcher(doc)
        matched_spans = []
        for match_id, start, end in matches:
            matched_spans.append(Span(doc, start, end))
        attrs = {}
        for k, v in self.patterns.items():
            if k not in ["patterns"]:
                attrs[k] = v
        group = SpanGroup(doc, spans=matched_spans, attrs=attrs)
        return SpanMatcher.keep_longest(group) if keep_longest else group


# ----------------------------------------------------------------------------------------------------------------------
@Language.factory(
    "span_match_ruler",
    default_config={"keep_longest": True},
)
class SpanRuler:
    def __init__(self, nlp: Language, name: str, keep_longest):
        self.name: str = name
        self.keep_longest = keep_longest
        self.matchers: List[SpanMatcher] = []

    def add(self, name: str, patterns: Dict):
        self.matchers.append(SpanMatcher(name, patterns))

    def clear(self):
        self.matchers = []

    def __call__(self, doc):
        for matcher in self.matchers:
            group = matcher.match(doc, self.keep_longest)
            doc.spans[matcher.name] = group
        return doc
