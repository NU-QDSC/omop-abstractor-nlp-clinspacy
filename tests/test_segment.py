from clinspacy import parse


def test_parse_section_patterns(suggest_request, abstractor):
    section_0 = suggest_request.abstractor_sections[0]
    name, patterns = parse.parse_section(section_0, abstractor.nlp)
    assert name == "SPECIMEN"
    assert len(patterns) == 1

    section_1 = suggest_request.abstractor_sections[1]
    name, patterns = parse.parse_section(section_1, abstractor.nlp)
    assert name == "COMMENT"
    assert len(patterns) == 6


def test_detect_sections(abstractor, notes, suggest_request):
    for section in suggest_request.abstractor_sections:
        name, patterns = parse.parse_section(section, abstractor.nlp)
        abstractor.sectionizer.add_patterns(name, patterns)

    doc = abstractor.nlp(notes[0])
    assert len(doc.spans["section_headers"]) == 10
    assert len(doc.spans["section_headers"].attrs["names"]) == 10
    assert len(doc.spans["sections"]) == 10
