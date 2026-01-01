from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema import Document
from pathlib import Path

DATA_DIR = Path("data/raw")

def load_papers():
    documents = []
    for pdf_path in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_path.name
            page.metadata["domain"] = "knowledge_distillation_object_detection"
            documents.append(page)

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,        # small enough for precision
        chunk_overlap=120,     # preserve context
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    # Add chunk IDs
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks
