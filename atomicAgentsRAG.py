from typing import List, Dict, Set
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


def load_pdfs(pdf_paths: List[str]) -> Dict[str, List[str]]:
    """
    Loads text content from a list of PDF files.

    Args:
        pdf_paths (List[str]): List of paths to PDF files.

    Returns:
        Dict[str, List[str]]: Dictionary mapping filenames to extracted text pages.
    """
    pdf_texts = {}
    for path in pdf_paths:
        try:
            reader = PdfReader(path)
            texts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            pdf_texts[os.path.basename(path)] = texts
            print(f"[INFO] Loaded {len(texts)} pages from {path}")
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {path}: {e}")
    return pdf_texts

def chunk_texts(pdf_texts: Dict[str, List[str]], chunk_size: int = 500) -> Dict[str, List[str]]:
    """
    Splits text from PDFs into smaller chunks for embedding.

    Args:
        pdf_texts (Dict[str, List[str]]): Dictionary of PDF filename -> list of texts.
        chunk_size (int): Maximum number of characters per chunk.

    Returns:
        Dict[str, List[str]]: Dictionary mapping filenames to text chunks.
    """
    chunked = {}
    for pdf_name, texts in pdf_texts.items():
        chunks = []
        for text in texts:
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
        chunked[pdf_name] = chunks
        print(f"[INFO] Created {len(chunks)} chunks from {pdf_name}")
    return chunked


### Embedding Engine with FAISS ###
class EmbeddingEngine:
    """
    Handles embedding creation and similarity search using FAISS.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.texts: List[str] = []
        self.sources: List[str] = []

    def build_index(self, chunked_texts: Dict[str, List[str]]):
        """
        Builds a FAISS index from all chunks.

        Args:
            chunked_texts (Dict[str, List[str]]): PDF name -> list of text chunks.
        """
        all_chunks, all_sources = [], []
        for pdf_name, chunks in chunked_texts.items():
            all_chunks.extend(chunks)
            all_sources.extend([pdf_name] * len(chunks))

        self.texts = all_chunks
        self.sources = all_sources

        if not all_chunks:
            raise ValueError("No text chunks provided for indexing.")

        embeddings = self.embedder.encode(all_chunks, convert_to_numpy=True).astype(np.float32)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"[INFO] FAISS index built with {len(all_chunks)} chunks (dimension: {dim})")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Retrieves the most similar text chunks for a given query.

        Args:
            query (str): Search query.
            top_k (int): Number of chunks to retrieve.

        Returns:
            List[Dict[str, str]]: List of retrieved chunks and their source PDFs.
        """
        if self.index is None:
            raise ValueError("Index not built. Please build the index first.")

        query_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        k = min(top_k, len(self.texts))
        distances, ids = self.index.search(query_emb, k)
        valid_ids = ids[0][ids[0] != -1]

        return [
            {"text": self.texts[i], "source": self.sources[i]}
            for i in valid_ids
        ]


### Pydantic Schemas ###
class RetrieveInput(BaseIOSchema):
    """Input schema for the Retrieval Agent."""
    query: str = Field(..., description="User query")


class RetrieveOutput(BaseIOSchema):
    """Output schema for the Retrieval Agent."""
    retrieved_chunks: List[Dict[str, str]] = Field(..., description="List of retrieved chunks and sources")


class AnswerInput(BaseIOSchema):
    """Input schema for the Answering Agent."""
    query: str = Field(..., description="User question")
    retrieved_chunks: List[Dict[str, str]] = Field(..., description="List of retrieved text chunks")


class AnswerOutput(BaseIOSchema):
    """Output schema for the Answering Agent."""
    answer: str = Field(..., description="Generated answer")
    sources: Set[str] = Field(..., description="Set of PDF filenames used as sources")


### System Prompt ###
system_prompt_generator = SystemPromptGenerator(
    background=["Medical RAG assistant specialized in clinical guidelines."],
    steps=[
        "Analyze the contents of the provided text chunks.",
        "Create a concise, fact-based summary that answers the user's question."
    ],
    output_instructions=[
        "Provide a precise and informative summary based solely on the given text chunks."
    ]
)

### Mistral Client ###
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_raw = Mistral(api_key=MISTRAL_API_KEY)
mistral_client = from_mistral(mistral_raw)

### Agents ###
class RetrieveAgent(AtomicAgent[RetrieveInput, RetrieveOutput]):
    def __init__(self, *, config: AgentConfig, embedding_engine: EmbeddingEngine, **kwargs):
        super().__init__(config=config, **kwargs)
        self.embedding_engine = embedding_engine

    def run(self, inputs: RetrieveInput) -> RetrieveOutput:
        chunks = self.embedding_engine.retrieve(inputs.query)
        return RetrieveOutput(retrieved_chunks=chunks)

class AnsweringAgent(AtomicAgent[AnswerInput, AnswerOutput]):
    """Agent that answers user questions based on retrieved document chunks."""

    def __init__(self, *, config: AgentConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.config = config

    def run(self, inputs: AnswerInput) -> AnswerOutput:
        system_prompt = self.config.system_prompt_generator.generate_prompt()

        if not inputs.retrieved_chunks:
            return AnswerOutput(answer="No relevant information found.", sources=set())

        chunks_text = "\n---\n".join(
            f"[Source: {chunk['source']}]\n{chunk['text']}"
            for chunk in inputs.retrieved_chunks
        )

        prompt = (
            f"The user asked the following question:\n\n"
            f"**{inputs.query}**\n\n"
            "Here are the most relevant text excerpts from medical guidelines.\n"
            "Use only this information to answer the question accurately.\n"
            "If the answer cannot be found, state that explicitly.\n\n"
            f"{chunks_text}"
        )

        response = self.config.client.chat.completions.create(
            model=self.config.model,
            response_model=AnswerOutput,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        sources = {chunk["source"] for chunk in inputs.retrieved_chunks}
        response.sources = sources
        return response


### Load documents, initialize agents and create the q&a loop in the main ###
if __name__ == "__main__":
    pdf_files = [
        "knowledge/leitlinie_atemwegsmanagement.pdf",
        "knowledge/leitlinie_schaedelhirntrauma.pdf",
        "knowledge/leitlinie_urtikaria.pdf"
    ]

    try:
        print("[INFO] Initializing system...")
        pdf_texts = load_pdfs(pdf_files)
        chunked_texts = chunk_texts(pdf_texts)

        embedding_engine = EmbeddingEngine()
        embedding_engine.build_index(chunked_texts)

        agent_config = AgentConfig(
            client=mistral_client,
            model="mistral-medium-latest",
            system_prompt_generator=system_prompt_generator,
        )

        retrieve_agent = RetrieveAgent(config=agent_config, embedding_engine=embedding_engine)
        answer_agent = AnsweringAgent(config=agent_config)

        print("\n[READY] The system is ready for questions. Type 'exit' to quit.\n")

        while True:
            question = input("Enter your question: ").strip()
            if question.lower() in {"exit", "quit"}:
                print("[INFO] Exiting...")
                break

            retrieved = retrieve_agent.run(RetrieveInput(query=question))
            answer = answer_agent.run(AnswerInput(query=question, retrieved_chunks=retrieved.retrieved_chunks))
            
            print("\n" + "=" * 60)
            print("Answer:")
            print(answer.answer)
            print("\nSources:")
            for src in answer.sources:
                print(f"- {src}")
            print("=" * 60 + "\n")

    except Exception as e:
        print(f"[FATAL] An unexpected error occurred: {e}")

