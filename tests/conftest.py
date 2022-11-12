import os
import yaml
import json
import pytest
from clinspacy.abstract import TextAbstractor
from importlib_resources import files
import textabstractor_testdata.breast as data
from pathlib import Path
from typing import Dict, List
from textabstractor.dataclasses import (
    AbstractionSchema,
    SuggestRequest,
    SuggestionSet,
    ProcessTextResponse,
)


dir_path = Path(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(scope="session")
def config():
    with open(dir_path / "config.yml") as conf_file:
        conf = yaml.load(conf_file, Loader=yaml.FullLoader)
        return conf


@pytest.fixture(scope="session")
def notes() -> List[str]:
    paths = [
        dir_path / "data/breast/note-1-text.txt",
        dir_path / "data/breast/note-2-text.txt",
        dir_path / "data/breast/note-3-text.txt",
        dir_path / "data/breast/note-4-text.txt",
    ]
    return [path.read_text() for path in paths]


@pytest.fixture(scope="session")
def responses() -> List[ProcessTextResponse]:
    responses = []
    paths = [
        dir_path / "data/breast/note-1-response.json",
        dir_path / "data/breast/note-2-response.json",
        dir_path / "data/breast/note-3-response.json",
        dir_path / "data/breast/note-4-response.json",
    ]
    for path in paths:
        json_dict = json.loads(path.read_text())
        responses.append(ProcessTextResponse(**json_dict))
    return responses


@pytest.fixture(scope="function")
def suggest_request() -> SuggestRequest:
    json_dict = json.loads(files(data).joinpath("request.json").read_text())
    return SuggestRequest(**json_dict)


@pytest.fixture(scope="function")
def suggestion_set() -> SuggestionSet:
    json_dict = json.loads(files(data).joinpath("suggestion_set.json").read_text())
    return SuggestionSet(**json_dict)


@pytest.fixture(scope="function")
def schemas() -> Dict[int, AbstractionSchema]:
    schemas = {}
    for i in range(285, 300):
        json_dict = json.loads(files(data).joinpath(f"{str(i)}.json").read_text())
        schemas[i] = AbstractionSchema(**json_dict["abstractor_abstraction_schema"])
    return schemas


@pytest.fixture(scope="function")
def process_text_responses() -> Dict[str, ProcessTextResponse]:
    responses = {}
    predicates = [
        "has_cancer_histology",
        "has_cancer_site",
        "has_surgery_date",
        "pathological_tumor_staging_category",
    ]
    for pred in predicates:
        json_dict = json.loads(
            files(data).joinpath(f"process_text_{pred}.json").read_text()
        )
        responses[pred] = ProcessTextResponse(**json_dict)
    return responses


@pytest.fixture(scope="function")
def abstractor() -> TextAbstractor:
    return TextAbstractor()
