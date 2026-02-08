import os

project_name = "TriMind_RAG_Engine"

list_of_files = [
    f"{project_name}/__init__.py",

    # DATA
    f"{project_name}/data/__init__.py",
    f"{project_name}/data/pdfs/.gitkeep",
    f"{project_name}/data/txts/.gitkeep",
    f"{project_name}/data/processed/.gitkeep",

    # CONFIG
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/config.py",

    # PROMPTS
    f"{project_name}/prompts/__init__.py",
    f"{project_name}/prompts/rag_prompt_template.py",
    f"{project_name}/prompts/query_analysis_prompt.py",

    # UTILS
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/document_loader.py",
    f"{project_name}/utils/preprocessing.py",
    f"{project_name}/utils/embeddings.py",
    f"{project_name}/utils/vector_store.py",
    f"{project_name}/utils/evaluation.py",

    # MEMORY
    f"{project_name}/memory/__init__.py",
    f"{project_name}/memory/conversation_memory.py",

    # CHAINS
    f"{project_name}/chains/__init__.py",
    f"{project_name}/chains/retrieval_chain.py",
    f"{project_name}/chains/conversational_chain.py",
    f"{project_name}/chains/rag_pipeline.py",

    # TESTS
    f"{project_name}/tests/__init__.py",
    f"{project_name}/tests/test_memory.py",
    f"{project_name}/tests/test_retrieval.py",
    f"{project_name}/tests/test_performance.py",

    # ROOT FILES
    f"{project_name}/main.py",
    "requirements.txt",
    ".env",
    ".gitignore",
    "README.md",
]

for file_path in list_of_files:
    dir_name = os.path.dirname(file_path)

    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            pass

print("Project structure created successfully!")
