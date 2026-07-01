import logging
import os
import uuid
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("pdf_ingestion")

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://944f67cc-5393-49f0-80e5-9d69bfa5f793.eu-central-1-0.aws.cloud.qdrant.io",
)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "college_docs")

PDF_FOLDER = os.getenv("PDF_FOLDER", "XAVIER info")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
UPLOAD_BATCH_SIZE = int(os.getenv("UPLOAD_BATCH_SIZE", "100"))
SHOW_EMBEDDING_PROGRESS = os.getenv("SHOW_EMBEDDING_PROGRESS", "true").lower() == "true"

POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"{QDRANT_URL}/{COLLECTION_NAME}")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={"batch_size": EMBEDDING_BATCH_SIZE},
    show_progress=SHOW_EMBEDDING_PROGRESS,
)
embedding_dimension = len(embedding_model.embed_query("dimension probe"))

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n",". ","? ","! ","; ",", "," ",""],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str

@dataclass(frozen=True)
class ChunkRecord:
    text: str
    chunk_index: int
    page_number: int | None

def get_vector_params(collection_info) -> VectorParams:
    vectors = collection_info.config.params.vectors
    if isinstance(vectors, dict):
        if len(vectors) != 1:
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' uses named vectors. "
                "This ingestion script expects a single unnamed vector."
            )
        return next(iter(vectors.values()))
    return vectors


def is_cosine_distance(distance) -> bool:
    distance_value = getattr(distance, "value", distance)
    return str(distance_value).lower() == str(Distance.COSINE.value).lower()


def ensure_collection_exists() -> None:
    """Create the Qdrant collection or validate that the existing one is compatible."""
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_dimension,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "Created collection '%s' with vector_size=%s distance=%s",
            COLLECTION_NAME,
            embedding_dimension,
            Distance.COSINE,
        )
        return

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    vector_params = get_vector_params(collection_info)

    if vector_params.size != embedding_dimension:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' has vector size {vector_params.size}, "
            f"but model '{EMBEDDING_MODEL_NAME}' produces {embedding_dimension}. "
            "Recreate the collection or use a matching embedding model."
        )

    if not is_cosine_distance(vector_params.distance):
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' uses distance {vector_params.distance}, "
            f"but this ingestion script requires {Distance.COSINE}. "
            "Recreate the collection with cosine distance before uploading."
        )

    logger.info(
        "Validated collection '%s' with vector_size=%s distance=%s",
        COLLECTION_NAME,
        vector_params.size,
        vector_params.distance,
    )


def extract_page_texts(pdf_path: str) -> list[PageText]:
    """Extract text from a PDF, preserving source page numbers."""
    reader = PdfReader(pdf_path)
    pages: list[PageText] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages.append(PageText(page_number=index, text=text))

    return pages


def build_clean_text_with_page_ranges(pages):
    document_parts: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []
    cursor = 0

    for page in pages:
        if document_parts:
            document_parts.append("\n\n")
            cursor += 2

        start = cursor
        document_parts.append(page.text)
        cursor += len(page.text)
        page_ranges.append((start, cursor, page.page_number))

    full_text = "".join(document_parts)
    full_text = "".join(document_parts).strip()
    return full_text, page_ranges


def page_number_for_offset(offset: int, page_ranges: Sequence[tuple[int, int, int]]) -> int | None:
    for start, end, page_number in page_ranges:
        if start <= offset < end:
            return page_number
    return page_ranges[-1][2] if page_ranges else None


def chunk_document(full_text: str, page_ranges: Sequence[tuple[int, int, int]]) -> list[ChunkRecord]:
    chunks = splitter.split_text(full_text)
    records: list[ChunkRecord] = []
    search_start = 0

    for index, chunk in enumerate(chunks):
        start = full_text.find(chunk, search_start)
        if start == -1:
            start = full_text.find(chunk)
        if start == -1:
            start = search_start

        records.append(
            ChunkRecord(
                text=chunk,
                chunk_index=index,
                page_number=page_number_for_offset(start, page_ranges),
            )
        )
        search_start = max(start + 1, search_start)

    return records


def generate_embeddings(texts: Sequence[str]):
    return embedding_model.embed_documents(list(texts))


def make_point_id(source: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{source}:{chunk_index}"))


def batched(items: Sequence[ChunkRecord], batch_size: int) -> Iterator[Sequence[ChunkRecord]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def upload_points(points: list[PointStruct]) -> None:
    if not points:
        return

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    logger.info("Uploaded batch with %s points", len(points))


def build_points(
    source: str,
    chunks: Sequence[ChunkRecord],
    embeddings: Iterable[Sequence[float]],
    total_chunks: int,
) -> list[PointStruct]:
    points: list[PointStruct] = []

    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=make_point_id(source, chunk.chunk_index),
                vector=list(embedding),
                payload={
                    "text": chunk.text,
                    "source": source,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "embedding_model": EMBEDDING_MODEL_NAME,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "total_chunks": total_chunks,
                },
            )
        )

    return points


def process_pdf(pdf_path: str) -> int:
    source = os.path.basename(pdf_path)
    logger.info("Processing PDF '%s'", source)

    try:
        pages = extract_page_texts(pdf_path)
    except Exception:
        logger.exception("Skipping unreadable PDF '%s'", source)
        return 0

    logger.info("Extracted text from %s pages in '%s'", len(pages), source)
    if not pages:
        logger.warning("Skipping '%s' because no text was extracted", source)
        return 0

    full_text, page_ranges = build_clean_text_with_page_ranges(pages)
    if not full_text:
        logger.warning("Skipping '%s' because no text remained after cleaning", source)
        return 0

    chunks = chunk_document(full_text, page_ranges)
    logger.info("Created %s chunks from '%s'", len(chunks), source)

    uploaded_count = 0
    for chunk_batch in batched(chunks, UPLOAD_BATCH_SIZE):
        texts = [chunk.text for chunk in chunk_batch]
        embeddings = generate_embeddings(texts)
        points = build_points(source, chunk_batch, embeddings, total_chunks=len(chunks))
        upload_points(points)
        uploaded_count += len(points)

    return uploaded_count


def iter_pdf_paths(pdf_folder: str) -> Iterator[str]:
    for filename in sorted(os.listdir(pdf_folder)):
        if filename.lower().endswith(".pdf"):
            yield os.path.join(pdf_folder, filename)


def main() -> None:
    # Delete old collection
    if client.collection_exists(collection_name=COLLECTION_NAME):
        logger.info("Deleting existing collection '%s'", COLLECTION_NAME)
        client.delete_collection(collection_name=COLLECTION_NAME)

    # Create fresh collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_dimension,
            distance=Distance.COSINE,
        ),
    )

    total_uploaded = 0
    for pdf_path in iter_pdf_paths(PDF_FOLDER):
        total_uploaded += process_pdf(pdf_path)

    logger.info("Uploaded %s chunks to Qdrant", total_uploaded)


if __name__ == "__main__":
    main()
