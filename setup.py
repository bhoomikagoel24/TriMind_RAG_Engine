from setuptools import setup, find_packages

with open("README.md","r", encoding="utf-8") as fp:
    long_description = fp.read()

# Editing the below variables as per our requirements
REPO_NAME = "TriMind_RAG_Engine"
AUTHOR_USER_NAME = "Bhoomika Goel"
SRC_REPO = "TriMind_RAG_Engine"

# Read dependencies
with open("requirements.txt", "r") as f:
    LIST_OF_REQUIREMENTS = f.read().splitlines()

setup(
    name=SRC_REPO,
    version="0.0.1",
    author="Bhoomika Goel",
    description="Scalable and Modular RAG Pipeline with Vector Store, Memory, and Observability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author_email="bhoomikagoel24@gmail.com",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)
