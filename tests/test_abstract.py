import pytest
import textabstractor
from clinspacy import abstract
from pathlib import Path
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


@pytest.mark.smoke
def test_pipeline_loaded(abstractor):
    assert abstractor.nlp is not None
    assert abstractor.nlp.pipe_names == [
        "pysbd",
        "sectionizer",
        "tagger",
        "attribute_ruler",
        "lemmatizer",
        "span_match_ruler",
        "negex",
        "relextractor",
    ]


@pytest.mark.parametrize(
    "note_number, expected",
    [
        (0, (7450, 1669, 148)),
        (1, (11316, 2363, 192)),
        (2, (1691, 335, 24)),
    ],
)
def test_sentence_splits(abstractor, suggest_request, notes, note_number, expected):
    assert len(suggest_request.abstractor_abstraction_schemas) == 15
    assert len(suggest_request.abstractor_sections) == 2

    suggest_request.text = notes[note_number]
    doc = abstractor.nlp(suggest_request.text)
    assert len(suggest_request.text) == expected[0]
    assert len(doc) == expected[1]
    assert len(list(doc.sents)) == expected[2]


@pytest.mark.parametrize("note_number", [0, 1, 2, 3])
def test_extract_suggestions(suggest_request, schemas, notes, responses, note_number):
    for schema_meta_data in suggest_request.abstractor_abstraction_schemas:
        textabstractor.textabstract.schema_cache[
            schema_meta_data.abstractor_abstraction_schema_uri
        ] = (
            schema_meta_data,
            schemas[schema_meta_data.abstractor_abstraction_schema_id],
        )

    suggest_request.text = notes[note_number]
    response = abstract.process_text(suggest_request)
    # assert response.sections == responses[note_number].sections
    assert response.sentences == responses[note_number].sentences
    assert len(response.suggestions) == len(responses[note_number].suggestions)

    # print(">"*80)
    # print(f"note number: {note_number}")
    # print(notes[note_number])
    # for s in response.suggestions:
    #     if s not in responses[note_number].suggestions:
    #         print("="*80)
    #         print("extra in actual response")
    #         print("="*80)
    #         print(s)
    # for s in responses[note_number].suggestions:
    #     if s not in response.suggestions:
    #         print("="*80)
    #         print("missing from actual response")
    #         print("="*80)
    #         print(s)
    # print("<"*80)
    # Path(f"./data/breast/note-{note_number+1}-response.json").write_bytes(
    #     JSONResponse(content=jsonable_encoder(response)).body
    # )


def test_extract_name_value_suggestions(suggest_request, schemas):
    for schema_meta_data in suggest_request.abstractor_abstraction_schemas:
        textabstractor.textabstract.schema_cache[
            schema_meta_data.abstractor_abstraction_schema_uri
        ] = (
            schema_meta_data,
            schemas[schema_meta_data.abstractor_abstraction_schema_id],
        )

    suggest_request.text = """
    A. The tumor size was 1cm at the greatest extent.
    B. HER2 FISH is POSITIVE.
    C. HER2 FISH is NEGATIVE.
    Note: The specimen was collected on 10/13/1968.
    """
    response = abstract.process_text(suggest_request)
    assert len(response.sections) == 4
    assert len(response.sentences) == 5
    assert len(response.suggestions) == 8

    assert (
        len(
            [
                s
                for s in response.suggestions
                if s.predicate == "has_her2_status"
                and s.type == "name"
                and s.value == "her2"
            ]
        )
        == 2
    )
    assert (
        len(
            [
                s
                for s in response.suggestions
                if s.predicate == "has_her2_status"
                and s.type == "value"
                and s.value == "positive"
            ]
        )
        == 1
    )
    assert (
        len(
            [
                s
                for s in response.suggestions
                if s.predicate == "has_her2_status"
                and s.type == "value"
                and s.value == "negative"
            ]
        )
        == 1
    )


def test_overlapping_name_values(suggest_request, schemas):
    suggest_request.abstractor_abstraction_schemas = [
        s
        for s in suggest_request.abstractor_abstraction_schemas
        if s.abstractor_abstraction_schema_id in [293, 294]
    ]
    suggest_request.abstractor_abstraction_schemas = [
        m
        for m in suggest_request.abstractor_abstraction_schemas
        if m.abstractor_abstraction_schema_id in [293, 294]
    ]
    for schema_metadata in suggest_request.abstractor_abstraction_schemas:
        textabstractor.textabstract.schema_cache[
            schema_metadata.abstractor_abstraction_schema_uri
        ] = (
            schema_metadata,
            schemas[schema_metadata.abstractor_abstraction_schema_id],
        )

    suggest_request.text = """
    TNM Staging: ypT2, N1a
    """
    response = abstract.process_text(suggest_request)
    assert len(response.suggestions) == 4
    assert (
        len(
            [
                s
                for s in response.suggestions
                if s.predicate == "pathological_tumor_staging_category"
                and s.type == "name"
            ]
        )
        == 1
    )
    assert (
        len(
            [
                s
                for s in response.suggestions
                if s.predicate == "pathological_tumor_staging_category"
                and s.type == "value"
            ]
        )
        == 1
    )
    assert (
        len(
            [
                s
                for s in response.suggestions
                if s.predicate == "pathological_nodes_staging_category"
                and s.type == "name"
            ]
        )
        == 1
    )
    assert (
        len(
            [
                s
                for s in response.suggestions
                if s.predicate == "pathological_nodes_staging_category"
                and s.type == "value"
            ]
        )
        == 1
    )


def test_negated_values(suggest_request, schemas):
    for schema_metadata in suggest_request.abstractor_abstraction_schemas:
        textabstractor.textabstract.schema_cache[
            schema_metadata.abstractor_abstraction_schema_uri
        ] = (
            schema_metadata,
            schemas[schema_metadata.abstractor_abstraction_schema_id],
        )

    suggest_request.text = """
    A. DCIS of the left breast was found.
    B. Invasive ductal carcinoma in the left breast was ruled out.
    C. Neuroendocrine cancer in the right breast was found,
    but neuroendocrine cancer in the right breast was ruled out.
    """
    response = abstract.process_text(suggest_request)
    histologies = [
        s for s in response.suggestions if s.predicate == "has_cancer_histology"
    ]
    sites = [s for s in response.suggestions if s.predicate == "has_cancer_site"]
    assert len(histologies) == 4
    assert len(sites) == 4
    assert histologies[0].assertion == "present"
    assert histologies[1].assertion == "present"
    assert histologies[2].assertion == "absent"
    assert histologies[3].assertion == "absent"
