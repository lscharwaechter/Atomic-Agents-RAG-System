from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
from pydantic import Field
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from mistralai import Mistral
from instructor import from_mistral
import os

from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator

class TextChunk(BaseIOSchema):
    """A text chunk from a PDF including source metadata (file & page number)."""
    text: str = Field(..., description="Chunk text")
    source_pdf: str = Field(..., description="PDF filename or path")
    page: int = Field(..., description="1-based page number in the PDF")

### PDF Loading & Chunking ###
def load_pdfs_with_pages(pdf_paths: List[str]) -> List[TextChunk]:
    """
    Loads PDFs and returns page-level TextChunks (not yet chunked),
    but already including source_pdf and page metadata.
    """
    pages: List[TextChunk] = []
    for path in pdf_paths:
        try:
            reader = PdfReader(path)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append(TextChunk(text=text, source_pdf=path, page=page_idx + 1))
        except Exception as e:
            print(f"Error while loading PDF {path}: {e}")
    return pages

def chunk_pages(pages: List[TextChunk], chunk_size: int = 800, overlap: int = 120) -> List[TextChunk]:
    """
    Chunking per page to ensure page metadata remains correct.
    Overlap helps reduce sentence truncation.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: List[TextChunk] = []
    step = chunk_size - overlap

    for p in pages:
        txt = p.text
        for i in range(0, len(txt), step):
            chunk_text = txt[i:i + chunk_size].strip()
            if chunk_text:
                chunks.append(TextChunk(text=chunk_text, source_pdf=p.source_pdf, page=p.page))
    return chunks

def _short_pdf_name(path: str) -> str:
    # without pathlib, intentionally simple:
    return path.split("/")[-1].split("\\")[-1]

def relevance_bucket(score: float) -> str:
    """Maps similarity score to a coarse confidence category."""
    if score >= 0.8:
        return "very relevant"
    if score >= 0.4:
        return "medium"
    return "rather irrelevant"

### Embedding Engine with FAISS ###
class EmbeddingEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[TextChunk] = []
        self._embeddings: Optional[np.ndarray] = None

    def build_index(self, chunks: List[TextChunk]):
        self.chunks = chunks
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True).astype(np.float32)
        self._embeddings = embeddings

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"FAISS index created with {len(chunks)} chunks. Dimension: {dim}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[TextChunk, float]]:
        """
        Returns (chunk, score).
        Note: IndexFlatL2 returns L2 distances -> smaller is better.
        We convert it into a simple similarity score: sim = 1 / (1 + dist)
        (only for display/heuristics; ranking still uses distance).
        """
        if self.index is None:
            return []

        query_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        k = min(top_k, len(self.chunks))
        distances, ids = self.index.search(query_emb, k)

        results: List[Tuple[TextChunk, float]] = []
        for dist, idx in zip(distances[0], ids[0]):
            if idx == -1:
                continue
            sim = float(1.0 / (1.0 + dist))
            results.append((self.chunks[int(idx)], sim))
        return results

### Pydantic Schemas ###
class RetrieveInput(BaseIOSchema):
    """Input schema for the retrieval agent. Contains the user's search query."""
    query: str = Field(..., description="Search query")

class RetrieveOutput(BaseIOSchema):
    """Output schema of the retrieval agent. Contains found text chunks (with metadata) and relevance scores."""
    retrieved: List[TextChunk] = Field(..., description="Retrieved text chunks")

    top_score: float = Field(..., description="Highest similarity score among top-k chunks (0..1, higher is better)")
    mean_score: float = Field(..., description="Average similarity score among top-k chunks (0..1, higher is better)")

    top_relevance: str = Field(..., description="Bucket category for top_score")
    mean_relevance: str = Field(..., description="Bucket category for mean_score")

class AnswerInput(BaseIOSchema):
    """Input schema for the answering agent. Contains question + relevant chunks + retrieval confidence."""
    query: str = Field(..., description="Original user question")
    retrieved_chunks: List[TextChunk] = Field(..., description="List of text chunks (with metadata)")
    top_score: float = Field(..., description="Top similarity score")
    mean_score: float = Field(..., description="Mean similarity score")
    top_relevance: str = Field(..., description="Bucket for top score")
    mean_relevance: str = Field(..., description="Bucket for mean score")

class AnswerOutput(BaseIOSchema):
    """Output schema for the answering agent. Contains the final answer including source list."""
    summary: str = Field(..., description="Generated answer")

### System Prompt ###
system_prompt_generator = SystemPromptGenerator(
    background=["Medical RAG assistant specialized in clinical guidelines."],
    steps=[
        "Analyze the content of the provided text chunks.",
        "Answer the question in a fact-based and concise manner."
    ],
    output_instructions=[
        "Use exclusively information from the provided text chunks.",
        "Every medical statement must end with a source reference in parentheses, e.g. (Q1, p.12).",
        "If the information is not contained in the chunks, explicitly state this."
    ]
)

### Mistral Client & Instructor Wrapper ###
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_raw = Mistral(api_key=MISTRAL_API_KEY)
mistral_client = from_mistral(mistral_raw)

### Helper Functions for Source Mapping ###
def build_q_mapping(chunks: List[TextChunk]) -> Tuple[Dict[str, str], List[str]]:
    """
    Creates a stable mapping per answer: source_pdf -> Q1..Qn (order of appearance).
    Returns mapping and a source list (strings).
    """
    mapping: "OrderedDict[str, str]" = OrderedDict()
    q_counter = 1
    for c in chunks:
        src = c.source_pdf
        if src not in mapping:
            mapping[src] = f"Q{q_counter}"
            q_counter += 1

    sources_lines = [f"{qid}: {_short_pdf_name(src)}" for src, qid in mapping.items()]
    return dict(mapping), sources_lines

def format_context_for_llm(chunks: List[TextChunk], qmap: Dict[str, str]) -> str:
    """
    Formats chunks so the LLM can easily cite inline.
    """
    formatted = []
    for i, c in enumerate(chunks, start=1):
        qid = qmap[c.source_pdf]
        # Important: we provide the citation format exactly as it should appear in the text
        header = f"Chunk {i} ({qid}, p.{c.page}):"
        formatted.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(formatted)

### Agents ###
class RetrieveAgent(AtomicAgent[RetrieveInput, RetrieveOutput]):
    def __init__(self, *, config: AgentConfig, embedding_engine: EmbeddingEngine, top_k: int = 5, **kwargs):
        super().__init__(config=config, **kwargs)
        self.embedding_engine = embedding_engine
        self.top_k = top_k

    def run(self, inputs: RetrieveInput) -> RetrieveOutput:
        retrieved = self.embedding_engine.retrieve(inputs.query, top_k=self.top_k)

        chunks_only = [c for (c, _sim) in retrieved]
        sims = [sim for (_c, sim) in retrieved]

        # If empty for any reason:
        if not sims:
            return RetrieveOutput(
                retrieved=[],
                top_score=0.0,
                mean_score=0.0,
                top_relevance=relevance_bucket(0.0),
                mean_relevance=relevance_bucket(0.0),
            )

        top_score = float(max(sims))
        mean_score = float(sum(sims) / len(sims))

        return RetrieveOutput(
            retrieved=chunks_only,
            top_score=top_score,
            mean_score=mean_score,
            top_relevance=relevance_bucket(top_score),
            mean_relevance=relevance_bucket(mean_score),
        )

class AnswerAgent(AtomicAgent[AnswerInput, AnswerOutput]):
    def __init__(self, *, config: AgentConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.config = config

    def run(self, inputs: AnswerInput) -> AnswerOutput:
        system_prompt = self.config.system_prompt_generator.generate_prompt()

        if not inputs.retrieved_chunks:
            return AnswerOutput(summary="No relevant document sections found to answer this question.")

        # Q-mapping per answer
        qmap, sources_lines = build_q_mapping(inputs.retrieved_chunks)
        context = format_context_for_llm(inputs.retrieved_chunks, qmap)

        user_prompt = (
            f"Question:\n{inputs.query}\n\n"
            "Context (chunks from guidelines, each with citation ID):\n"
            f"{context}\n\n"
            "Task:\n"
            "- Answer the question exclusively based on the context.\n"
            "- Add a source reference after each medical statement in the form (Qx, p.y).\n"
            "- Use the Q-IDs and page numbers exactly as given in the context.\n"
            "- If something is not supported, write: 'I cannot find reliable information on this in the provided guideline excerpts.'\n"
        )

        response = self.config.client.chat.completions.create(
            model=self.config.model,
            response_model=AnswerOutput,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        # Append source list + confidence
        sources_block = "Sources:\n" + "\n".join(sources_lines)

        confidence_block = (
            "Confidence (embedding-based):\n"
            f"- Top-chunk relevance = {inputs.top_relevance} (Score={inputs.top_score:.3f})\n"
            f"- Mean relevance      = {inputs.mean_relevance} (Score={inputs.mean_score:.3f})"
        )

        merged = response.summary.strip() + "\n\n" + sources_block + "\n\n" + confidence_block
        return AnswerOutput(summary=merged)

class RAGPipeline:
    def __init__(self, pdf_paths: List[str]):
        print("Starting RAG pipeline initialization...")
        pages = load_pdfs_with_pages(pdf_paths)
        chunks = chunk_pages(pages, chunk_size=800, overlap=120)

        if not chunks:
            raise ValueError("No text chunks extracted from PDFs. Check paths and content.")

        self.embedding_engine = EmbeddingEngine()
        self.embedding_engine.build_index(chunks)
        print("Indexing completed.")

        agent_config = AgentConfig(
            client=mistral_client,
            model="mistral-medium-latest",
            system_prompt_generator=system_prompt_generator,
        )

        self.retrieve_agent = RetrieveAgent(config=agent_config, embedding_engine=self.embedding_engine, top_k=5)
        self.answer_agent = AnswerAgent(config=agent_config)

        print("Agents initialized.")

    def ask(self, question: str) -> AnswerOutput:
        print(f"\n-> Question: {question}")
        print("1. Retrieval...")
        retrieved = self.retrieve_agent.run(RetrieveInput(query=question))
        print(f"   Retrieved chunks: {len(retrieved.retrieved)}")

        if not retrieved.retrieved:
            return AnswerOutput(summary="No relevant information could be found in the documents.")

        print("2. Answering (agent-based)...")
        summary = self.answer_agent.run(
            AnswerInput(
                query=question,
                retrieved_chunks=retrieved.retrieved,
                top_score=retrieved.top_score,
                mean_score=retrieved.mean_score,
                top_relevance=retrieved.top_relevance,
                mean_relevance=retrieved.mean_relevance,
            )
        )
        return summary

### Load documents, initialize agents and answer the question ###
if __name__ == "__main__":
    pdf_files = [
        "knowledge/leitlinie_atemwegsmanagement.pdf",
        "knowledge/leitlinie_schaedelhirntrauma.pdf",
        "knowledge/leitlinie_urtikaria.pdf"
    ]

    try:
        pipeline = RAGPipeline(pdf_files)
        question = "Which indications are there for hospital admission in traumatic brain injury?"
        result = pipeline.ask(question)

        print("\n" + "="*50)
        print("Answer (with sources):")
        print(result.summary)
        print("="*50)

    except Exception as e:

        print(f"\n[ERROR] An error occurred: {e}")

