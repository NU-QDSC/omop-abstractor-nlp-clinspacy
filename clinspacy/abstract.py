import spacy
from spacy.symbols import ORTH
import textabstractor
from contextlib import contextmanager
from pluggy import HookimplMarker
from clinspacy.match import *  # noqa: F401
from clinspacy.negate import Negex  # noqa: F401
from clinspacy.parse import *  # noqa: F401
from clinspacy.segment import *  # noqa: F401
from clinspacy.extract import *  # noqa: F401
from textabstractor.dataclasses import (
    SuggestRequest,
    ProcessTextResponse,
    AbstractionSchemaMetaData,
    SectionSpan,
    SentenceSpan,
    Suggestion,
)

# --------------------------------------------------------------------------------------------------
hookimpl = HookimplMarker(textabstractor.__project_name__)


# --------------------------------------------------------------------------------------------------
class TextAbstractor:
    def __init__(self):
        self.nlp = spacy.load(
            "en_core_web_sm", exclude=["parser", "tok2vec", "senter", "ner"]
        )
        self.sentencer = self.nlp.add_pipe("pysbd", first=True)
        self.sectionizer = self.nlp.add_pipe(
            "sectionizer", after="pysbd", config={"newline_breaks": False}
        )
        self.span_ruler = self.nlp.add_pipe("span_match_ruler", after="lemmatizer")
        self.negex = self.nlp.add_pipe("negex", after="span_match_ruler")
        self.relextractor = self.nlp.add_pipe("relextractor", after="negex")

        # Add tokenization special cases
        self.nlp.tokenizer.add_special_case("in-", [{ORTH: "in"}, {ORTH: "-"}])
        self.nlp.tokenizer.add_special_case("-situ", [{ORTH: "-"}, {ORTH: "situ"}])

    def clear(self):
        self.span_ruler.clear()
        self.sectionizer.clear()


# --------------------------------------------------------------------------------------------------
# TODO: replace with TinyDB or sqlite
# TODO: add cache for compiled schemas in SpanRuler
schema_cache: Dict[str, Tuple[AbstractionSchemaMetaData, Tuple[Dict, List[Dict]]]] = {}


# --------------------------------------------------------------------------------------------------
@hookimpl
def process_text(request: SuggestRequest) -> ProcessTextResponse:
    with apply_nlp(request) as doc:
        sections = extract_sections(doc)
        sentences = extract_sentences(doc)
        suggestions = filter_out_covered(extract_suggestions(doc))
    return ProcessTextResponse(
        sections=sections, sentences=sentences, suggestions=suggestions
    )


# --------------------------------------------------------------------------------------------------
@contextmanager
def apply_nlp(request: SuggestRequest) -> Doc:
    try:
        abstractor = TextAbstractor()
        for section in request.abstractor_sections:
            abstractor.sectionizer.add_patterns(*parse_section(section, abstractor.nlp))
        for meta_schema in request.abstractor_abstraction_schemas:
            name_patterns, value_patterns = get_schema_patterns(abstractor, meta_schema)
            if len(name_patterns) > 0 and len(value_patterns) > 0:
                name_patterns["value_patterns"] = value_patterns
                abstractor.span_ruler.add(name_patterns["predicate"], name_patterns)
            else:
                for vp in value_patterns:
                    abstractor.span_ruler.add(vp["value"], vp)
        yield abstractor.nlp(request.text)
    finally:
        pass


# --------------------------------------------------------------------------------------------------
def get_schema_patterns(
    abstractor: TextAbstractor, schema_metadata: AbstractionSchemaMetaData
) -> Tuple[Dict, List[Dict]]:
    schema_uri = schema_metadata.abstractor_abstraction_schema_uri
    rule_type = schema_metadata.abstractor_rule_type
    key = f"{schema_uri}:{rule_type}"
    if key in schema_cache:
        m, patterns = schema_cache[key]
        if schema_metadata.updated_at <= m.updated_at:
            return patterns

    schema = textabstractor.textabstract.get_abstraction_schema(schema_metadata)
    patterns = parse_schema(schema, schema_metadata, abstractor.nlp)
    schema_cache[key] = (schema_metadata, patterns)
    return patterns


# --------------------------------------------------------------------------------------------------
def extract_sections(doc: Doc) -> List[SectionSpan]:
    sections = []
    section_headers = doc.spans["section_headers"]
    section_names = doc.spans["section_headers"].attrs["names"]
    for idx, sect in enumerate(doc.spans["sections"]):
        sections.append(
            SectionSpan(
                section_number=idx,
                section_name=section_names[idx],
                begin=sect.start_char,
                end=sect.end_char - 1,
                begin_header=section_headers[idx].start_char,
                end_header=section_headers[idx].end_char - 1,
            )
        )
    return sections


# --------------------------------------------------------------------------------------------------
def extract_sentences(doc: Doc) -> List[SentenceSpan]:
    sentences = []
    for idx, sent in enumerate(doc.sents):
        sentences.append(
            SentenceSpan(
                sentence_number=idx, begin=sent.start_char, end=sent.end_char - 1
            )
        )
    return sentences


# --------------------------------------------------------------------------------------------------
def extract_suggestions(doc: Doc) -> List[Suggestion]:
    suggestions = []
    for _, span_group in doc.spans.items():
        rule_type: str = span_group.attrs.get("rule_type", None)
        if rule_type not in ["name", "value"]:
            continue
        for span in span_group:
            # suggestions for name and stand-alone value matches
            suggestions.append(
                Suggestion(
                    predicate=span_group.attrs["predicate"],
                    begin=span.start_char,
                    end=span.end_char - 1,
                    type=span_group.attrs["rule_type"],
                    value=span_group.attrs["value"],
                    assertion="absent" if span._.negated else "present",
                )
            )
            # suggestions for value matches corresponding to name matches
            value_map: Dict[Span, SpanGroup] = span_group.attrs.get("value_map", {})
            for value_span in value_map.get(span, []):
                suggestions.append(
                    Suggestion(
                        predicate=span_group.attrs["predicate"],
                        begin=value_span.start_char,
                        end=value_span.end_char - 1,
                        type="value",
                        value=value_span.label_,
                        assertion="present",
                    )
                )
    return suggestions


# --------------------------------------------------------------------------------------------------
def filter_out_covered(suggestions: List[Suggestion]) -> List[Suggestion]:
    return [
        suggestion
        for suggestion in suggestions
        if not [
            s
            for s in suggestions
            if s.begin <= suggestion.begin
            and s.end >= suggestion.end
            and len(s) > len(suggestion)
        ]
    ]
