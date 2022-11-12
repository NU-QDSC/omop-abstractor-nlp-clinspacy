import yaml
from typing import Tuple
from importlib_resources import files
from clinspacy import data
from clinspacy.match import *


config = yaml.safe_load(files(data).joinpath("config.yml").read_text())


@Language.factory(
    "negex",
    default_config={
        "pseudo_negations": config["negation"]["pseudo_negations"],
        "pre_negations": config["negation"]["pre_negations"],
        "post_negations": config["negation"]["post_negations"],
        "terminators": config["negation"]["terminators"],
    },
)
class Negex:
    def __init__(
        self,
        nlp: Language,
        name: str,
        pseudo_negations: List[str],
        pre_negations: List[str],
        post_negations: List[str],
        terminators: List[str],
    ):
        if not Span.has_extension("negated"):
            Span.set_extension("negated", default=False)

        self.name = name
        _ = Negex.parse_phrases("pseudo_negations", pseudo_negations, nlp)
        self.pseudo_neg_matcher = SpanMatcher(_["name"], _)
        _ = Negex.parse_phrases("pre_negations", pre_negations, nlp)
        self.pre_neg_matcher = SpanMatcher(_["name"], _)
        _ = Negex.parse_phrases("post_negations", post_negations, nlp)
        self.post_neg_matcher = SpanMatcher(_["name"], _)
        _ = Negex.parse_phrases("terminators", terminators, nlp)
        self.term_matcher = SpanMatcher(_["name"], _)

    @staticmethod
    def parse_phrases(name: str, texts: List[str], nlp: Language) -> Dict:
        pattern_map = {"name": name, "patterns": []}
        with nlp.select_pipes(enable=["tagger", "attribute_ruler", "lemmatizer"]):
            for text in texts:
                pattern = []
                doc = nlp(text)
                for token in doc:
                    pattern.append({"LOWER": token.text.lower()})
                pattern_map["patterns"].append(pattern)
        return pattern_map

    def find_negations(self, doc: Doc) -> Dict[str, SpanGroup]:
        pseudo_matches = self.pseudo_neg_matcher.match(doc, keep_longest=True)
        pre_matches = self.pre_neg_matcher.match(doc, keep_longest=True)
        post_matches = self.post_neg_matcher.match(doc, keep_longest=True)
        term_matches = self.term_matcher.match(doc, keep_longest=True)

        pre_matches = filter_covered(pseudo_matches, pre_matches)
        post_matches = filter_covered(pseudo_matches, post_matches)
        post_matches = filter_covered(pre_matches, post_matches)
        pre_matches = filter_covered(post_matches, pre_matches)

        return {
            "pseudo_negations": pseudo_matches,
            "pre_negations": pre_matches,
            "post_negations": post_matches,
            "terminators": term_matches,
        }

    @staticmethod
    def find_scopes(doc, span, terminators, **kwargs) -> Tuple[Span, Span]:
        start = 0
        end = len(doc)
        # if len(terminators) == 0:
        #     start = span.start
        for t in terminators:
            if span.start >= t.end > start:
                start = t.end
            elif span.end <= t.start < end:
                end = t.start
        return doc[start : span.start], doc[span.end : end]

    @staticmethod
    def neg_in_scope(scope: Span, negations: SpanGroup) -> bool:
        for neg in negations:
            if covers(scope, neg, strict=False):
                return True
        return False

    @staticmethod
    def aggregate_spans(doc: Doc) -> Dict[Span, List[Span]]:
        span_groups = [
            group
            for _, group in doc.spans.items()
            if group and group.attrs.get("rule_type", "") in ["value", "name"]
        ]
        spans = [span for group in span_groups for span in group]
        sents_to_spans: Dict[Span, List[Span]] = {}
        for span in spans:
            if span.sent in sents_to_spans:
                sents_to_spans[span.sent].append(span)
            else:
                sents_to_spans[span.sent] = [span]
        return sents_to_spans

    def __call__(self, doc):
        span_map = Negex.aggregate_spans(doc)
        for sent, spans in span_map.items():
            sent_doc = sent.doc[sent.start : sent.end].as_doc()
            neg_matches = self.find_negations(sent_doc)
            for span in spans:
                left_scope, right_scope = Negex.find_scopes(
                    sent_doc,
                    Span(sent_doc, span.start - sent.start, span.end - sent.start),
                    **neg_matches
                )
                if Negex.neg_in_scope(left_scope, neg_matches["pre_negations"]):
                    span._.negated = True
                elif Negex.neg_in_scope(right_scope, neg_matches["post_negations"]):
                    span._.negated = True
        return doc
