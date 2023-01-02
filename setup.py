from setuptools import setup, find_packages

# import textabstractor

ABOUT = {}
with open("clinspacy/about.py", "r") as about_file:
    exec(about_file.read(), ABOUT)

setup(
    name=ABOUT["__project_name__"],
    version=ABOUT["__version__"],
    author="Will Thompson",
    author_email="will.k.t@gmail.com",
    description="A full SpaCy pipeline and models for clinical documents.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=["clinical nlp spacy"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 1 - Alpha",
        "License :: MIT Software License",
    ],
    packages=find_packages(include=["clinspacy", "clinspacy.*"]),
    package_data={"clinspacy": ["data/*"]},
    include_package_data=True,
    entry_points={"textabstractor": ["clinspacy = clinspacy.abstract"]},
    python_requires=">=3.9.0",
    install_requires=[
        "textabstractor",
        "spacy>=3.4",
        "pluggy",
        "pysbd",
    ],
    extras_require={
        "interactive": ["jupyterlab", "rise"],
        "dev": [
            "black",
            "pyment",
            "twine",
            "tox",
            "bumpversion",
            "flake8",
            "coverage",
            "sphinx",
        ],
        "test": ["pytest", "starlette", "httpx"],
    },
)
