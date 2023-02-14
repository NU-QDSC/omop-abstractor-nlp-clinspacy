from typing import Dict, List
from spacy.matcher import Matcher
from spacy import Language
import pysbd
import re


@Language.factory("pysbd")
class PySBDSentenceSplitter:
    def __init__(self, name, nlp, clean=False):
        self.name = name
        self.nlp = nlp
        self.seg = pysbd.Segmenter(language="en", clean=clean, char_span=True)

    def __call__(self, doc):
        sents_char_spans = self.seg.segment(doc.text_with_ws)
        start_token_ids = [sent.start for sent in sents_char_spans]
        for token in doc:
            token.is_sent_start = True if token.idx in start_token_ids else False
        return doc


@Language.factory("sectionizer", default_config={"newline_breaks": False})
class Sectionizer:
    def __init__(self, name, nlp, newline_breaks: bool):
        self.name = name
        self.patterns: Dict[str, List] = {}
        self.section_breaks = []
        if newline_breaks:
            self.section_breaks.append(re.compile(r"\n{2,}"))

    def add_patterns(self, name, patterns):
        self.patterns[name] = patterns

    def clear(self):
        self.patterns = {}

    def find_section_break(self, doc: object, start: int, end: int) -> int:
        for t in doc[start:end]:
            for p in self.section_breaks:
                if p.match(t.text):
                    return t.i + 1
        return end

    def __call__(self, doc):
        doc.spans["section_headers"] = []
        doc.spans["section_headers"].attrs["names"] = []
        doc.spans["sections"] = []

        section_headers = doc.spans["section_headers"]
        section_header_names = doc.spans["section_headers"].attrs["names"]
        sections = doc.spans["sections"]

        if len(self.patterns) == 0:
            return doc

        for name, patterns in self.patterns.items():
            matcher_patterns = []
            regex_patterns = []
            for p in patterns:
                if 'REGEX_TEXT' in p:
                    regex_patterns.append(p['REGEX_TEXT'])
                else:
                    matcher_patterns.append(p)
            if regex_patterns:
                for p in regex_patterns:
                    for m in p.finditer(doc.text):
                        span = doc.char_span(m.start(1), m.end(1), alignment_mode="expand")
                        section_headers.append(span)
                        section_header_names.append(name)
            if matcher_patterns:
                matcher = Matcher(doc.vocab)
                matcher.add(name, matcher_patterns)
                matches = matcher(doc)
                for match_id, start, end in matches:
                    section_headers.append(doc[start:end])
                    section_header_names.append(name)

        sorted_section_headers = sorted(section_headers, key=lambda s: s.start)
        for idx, span in enumerate(sorted_section_headers):
            if idx + 1 < len(sorted_section_headers):
                end = sorted_section_headers[idx + 1].start
            else:
                end = len(doc)
            end = self.find_section_break(doc, span.start, end)
            sections.append(doc[span.start:end])

        return doc
