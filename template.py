import os

project_name = "TriMind_RAG_Engine"

list_of_files = [
    # ---------------- PACKAGE ---------------- #
    f"{project_name}/__init__.py",

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
    f"{project_name}/utils/document_processing.py",
    f"{project_name}/utils/evaluation.py",
    f"{project_name}/utils/cache.py",

    # MEMORY
    f"{project_name}/memory/__init__.py",
    f"{project_name}/memory/conversation_memory.py",
    f"{project_name}/memory/redis_memory.py", 

    # CHAINS
    f"{project_name}/chains/__init__.py",
    f"{project_name}/chains/retrieval_chain.py",
    f"{project_name}/chains/conversational_chain.py",
    f"{project_name}/chains/rag_pipeline.py",

    # ---------------- SERVICES ---------------- #
    f"{project_name}/services/__init__.py", 
    f"{project_name}/services/rag_service.py",
    f"{project_name}/services/ingestion_service.py",

    # ---------------- INTERFACES ---------------- #
    f"{project_name}/interfaces/__init__.py",  # NEW
    f"{project_name}/interfaces/cli.py", 
    f"{project_name}/interfaces/api/__init__.py",
    f"{project_name}/interfaces/api/app.py",
    f"{project_name}/interfaces/api/routes.py",
    
    # ---------------- LOGGING ---------------- #
    f"{project_name}/logging/__init__.py",
    f"{project_name}/logging/logger.py",

    # ---------------- EXCEPTION HANDLER ---------------- #
    f"{project_name}/exception_handler/__init__.py",
    f"{project_name}/exception_handler/custom_exception.py",

    # ---------------- ROOT LEVEL ---------------- #

    # "tests/__init__.py",
    # "tests/test_memory.py",
    # "tests/test_retrieval.py",
    # "tests/test_performance.py",
    "streamlit_app.py",
    "main.py",
    "requirements.txt",
    ".env",
    "setup.py",
    ".gitignore",
    "README.md",
]

for file_path in list_of_files:
    dir_name = os.path.dirname(file_path)

    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8"):
            pass

print("Project structure created successfully!")
