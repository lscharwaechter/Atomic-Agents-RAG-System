# RAG-System-using-Atomic-Agents
This example shows how to use the AI agent framework Atomic Agents to build an interactive Retrieval-Augmented Generation (RAG) system in Python. It loads a predefined set of PDFs, indexes their content using FAISS and SentenceTransformer, and answers user questions based on the given PDFs through the language model Mistral.

- Semantic retrieval over PDF documents using FAISS
- Chunking and embedding with SentenceTransformer
- Summarization and reasoning using the Mistral API
- Interactive Q&A loop in the console
- Source transparency - shows the PDFs that were used for the answer
- Modular design with agents (RetrieveAgent, SummarizeAgent)

Atomic Agents enables to define transparent communication between the agents using Pydantics input/output schemes. In this example, medical guideline PDFs are loaded to answer medical questions, such as "What are the indications for hospital admission in cases of traumatic brain injury?". The guidelines are downloaded from the platform for scientific medicine [AWMF](https://register.awmf.org/de/leitlinien/aktuelle-leitlinien/fachgesellschaft/008). This script is written for demonstration purposes only. For real-world applications, more sophisticated methods regarding information safety in critical scenarios as well as the generation of chunks based on the documents are required.

User Question &rarr; [RetrieveAgent] Finds relevant PDF text chunks using FAISS &rarr; [SummarizeAgent] Uses Mistral to summarize and answer based on retrieved chunks &rarr; Final Answer + Sources 
