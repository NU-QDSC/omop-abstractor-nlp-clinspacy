from spacy.language import Language
from spacy.tokens import Span, SpanGroup
from clinspacy.match import SpanMatcher


@Language.factory("relextractor")
class RelationExtractor:
    def __init__(self, nlp: Language, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self, doc):
        for name_group, value_patterns in [
            (group, group.attrs["value_patterns"])
            for _, group in doc.spans.items()
            if group.attrs.get("rule_type", "") == "name"
        ]:
            name_group.attrs["value_map"] = {}
            for vp in value_patterns:
                span_matcher = SpanMatcher(vp["predicate"], vp)
                for span in name_group:
                    # look for values on the right side of the span
                    right_span = doc[span.end : span.sent.end]
                    right_span_as_doc = right_span.as_doc()
                    value_group = span_matcher.match(
                        right_span_as_doc, keep_longest=True
                    )
                    if len(value_group) > 0:
                        group = name_group.attrs["value_map"].get(
                            span, SpanGroup(doc, spans=[])
                        )
                        for idx, s in enumerate(value_group):
                            group.append(
                                Span(
                                    doc,
                                    right_span.start + s.start,
                                    right_span.start + s.end,
                                    label=s.text
                                    if vp["value"] in ["date", "number"]
                                    else vp["value"],
                                )
                            )
                        name_group.attrs["value_map"][span] = group

                    # look for values on the left side of the span
                    left_span = doc[span.sent.start : span.start]
                    left_span_as_doc = left_span.as_doc()
                    value_group = span_matcher.match(
                        left_span_as_doc, keep_longest=True
                    )
                    if len(value_group) > 0:
                        group = name_group.attrs["value_map"].get(
                            span, SpanGroup(doc, spans=[])
                        )
                        for idx, s in enumerate(value_group):
                            group.append(
                                Span(
                                    doc,
                                    left_span.start + s.start,
                                    left_span.start + s.end,
                                    label=s.text
                                    if vp["value"] in ["date", "number"]
                                    else vp["value"],
                                )
                            )
                        name_group.attrs["value_map"][span] = group

        return doc
