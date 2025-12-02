import argparse
import os
from pathlib import Path
from typing import List, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class DocumentIngestor:
    """Carga documentos de reglas de negocio y genera un índice vectorial."""

    def __init__(
        self,
        data_dir: Path,
        vector_dir: Path,
        embedding_provider: str = "openai",
        embedding_model: Optional[str] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
    ) -> None:
        self.data_dir = data_dir
        self.vector_dir = vector_dir
        self.embedding_provider = embedding_provider.lower()
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _detect_documents(self) -> List[Path]:
        supported_extensions = {".pdf", ".docx", ".txt", ".md"}
        documents = [
            path
            for path in self.data_dir.rglob("*")
            if path.suffix.lower() in supported_extensions and path.is_file()
        ]
        if not documents:
            raise FileNotFoundError(
                f"No se encontraron documentos en {self.data_dir} con extensiones {supported_extensions}."
            )
        return documents

    def _load_documents(self, documents: List[Path]):
        loaded = []
        for path in documents:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
            elif path.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(path))
            else:
                loader = TextLoader(str(path), encoding="utf-8")
            loaded.extend(loader.load())
        return loaded

    def _embedding_model(self):
        if self.embedding_provider == "openai":
            return OpenAIEmbeddings(model=self.embedding_model or "text-embedding-3-large")
        if self.embedding_provider == "hf":
            return HuggingFaceEmbeddings(model_name=self.embedding_model or "all-MiniLM-L6-v2")
        raise ValueError(
            "Proveedor de embeddings no soportado. Use 'openai' o 'hf' (HuggingFace)."
        )

    def ingest(self) -> None:
        documents = self._load_documents(self._detect_documents())
        splitted = self.text_splitter.split_documents(documents)
        embeddings = self._embedding_model()
        vectorstore = FAISS.from_documents(splitted, embeddings)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.vector_dir))
        print(
            f"Índice creado con {len(splitted)} fragmentos. Guardado en: {self.vector_dir}"
        )


class CodeGenerator:
    """Genera fragmentos de código usando reglas de negocio indexadas."""

    def __init__(
        self,
        vector_dir: Path,
        embedding_provider: str = "openai",
        embedding_model: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
    ) -> None:
        self.vector_dir = vector_dir
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.model_name = model_name or "gpt-4o-mini"
        self.temperature = temperature

    def _embedding_model(self):
        if self.embedding_provider == "openai":
            return OpenAIEmbeddings(model=self.embedding_model or "text-embedding-3-large")
        if self.embedding_provider == "hf":
            return HuggingFaceEmbeddings(model_name=self.embedding_model or "all-MiniLM-L6-v2")
        raise ValueError("Proveedor de embeddings no soportado. Use 'openai' o 'hf'.")

    def _retriever(self):
        embeddings = self._embedding_model()
        vectorstore = FAISS.load_local(
            str(self.vector_dir), embeddings, allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    def _llm(self):
        return ChatOpenAI(temperature=self.temperature, model=self.model_name)

    def generate_code(self, requirement: str) -> str:
        retriever = self._retriever()
        llm = self._llm()
        template = PromptTemplate(
            input_variables=["context", "requirement"],
            template=(
                "Eres un asistente de desarrollo. Basado en las siguientes reglas de negocio, "
                "genera un fragmento de código claro y comentado que cumpla el requerimiento.\n"
                "Responde solo con el código y explica brevemente cada paso como comentarios.\n"
                "Contexto:\n{context}\n\n"
                "Requerimiento:\n{requirement}"
            ),
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": template},
            return_source_documents=False,
        )

        result = chain.invoke({"query": requirement})
        return result["result"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline LangChain para reglas de negocio y generación de código"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directorio donde se encuentran los documentos de reglas de negocio.",
    )
    parser.add_argument(
        "--vector_dir",
        type=Path,
        default=Path("vectores") / "prueba",
        help="Directorio donde se almacenará/cargará el índice FAISS.",
    )
    parser.add_argument(
        "--embedding_provider",
        choices=["openai", "hf"],
        default="openai",
        help="Proveedor de embeddings: openai u hf (HuggingFace).",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Modelo de embeddings a utilizar. Si no se indica se usan valores por defecto.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Modelo de chat para generación de código (por defecto gpt-4o-mini).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1200,
        help="Tamaño de fragmentos de texto para el indexado.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=100,
        help="Solapamiento entre fragmentos al dividir documentos.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingesta documentos y actualiza el índice vectorial.",
    )
    parser.add_argument(
        "--requirement",
        type=str,
        default=None,
        help="Requerimiento a resolver mediante generación de código.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperatura del modelo de generación.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.ingest:
        ingestor = DocumentIngestor(
            data_dir=args.data_dir,
            vector_dir=args.vector_dir,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        ingestor.ingest()

    if args.requirement:
        generator = CodeGenerator(
            vector_dir=args.vector_dir,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            model_name=args.model_name,
            temperature=args.temperature,
        )
        code = generator.generate_code(args.requirement)
        print("\n=== Fragmento generado ===\n")
        print(code)

    if not args.ingest and not args.requirement:
        print(
            "Debe indicar --ingest para crear el índice y/o --requirement para generar código."
        )


if __name__ == "__main__":
    # Claves API se leen de variables de entorno (OPENAI_API_KEY u otros proveedores)
    if not os.getenv("OPENAI_API_KEY") and os.getenv("EMBEDDING_PROVIDER", "openai") == "openai":
        print(
            "Advertencia: OPENAI_API_KEY no está definido. Configure la clave antes de usar OpenAI."
        )
    main()
