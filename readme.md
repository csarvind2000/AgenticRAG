# multi-agent Retrieval-Augmented Generation (RAG) with Neo4j Knowledge Graph

A multi-agent Retrieval-Augmented Generation (RAG) system designed to analyze PDFs. This implementation uses a Neo4j knowledge graph database instead of FAISS for semantic storage and retrieval, powered by LLaMA 3 via Ollama, and features a Gradio-based UI. Developed and tested on Linux.

## Features
- **PDF Upload**: Upload brochures via a Gradio web interface.
- **Text Extraction**: Extracts text from PDFs using `PyPDF2`.
- **Text Cleaner Agent**: Dynamically generates `re.sub` patterns with LLaMA 3 to remove noise and repetitions.
- **Knowledge Graph**: Stores entities and relationships in Neo4j for semantic querying.
- **Multi-Agent Workflow**: Includes agents for cleaning, analysis, Chain-of-Thought (CoT) reasoning, question generation, and answering.
- **Local Processing**: Runs LLaMA 3 via Ollama locally, no external APIs required.
- **Gradio UI**: Displays results in a user-friendly web interface.

## Prerequisites
1. **Linux OS**: Tested on Ubuntu/Debian; should work on other distributions.
2. **Python 3.8+**: Ensure installed (`python3 --version`).
3. **Ollama**:
   - Install via: `curl -fsSL https://ollama.com/install.sh | sh`
   - Pull LLaMA 3: `ollama pull llama3`
   - Run server: `ollama serve`
4. **Neo4j**:
   - Install via: `sudo apt-get install neo4j` (Ubuntu/Debian)
   - Start service: `sudo systemctl start neo4j`
   - Set password in Neo4j (default URI: `bolt://localhost:7687`).
5. **Hardware**: 16GB+ RAM recommended for LLaMA 3 inference.

## Output



    1. Cleaned Data: Text with noise removed.
    2. Analyzed Data: Concise summary of key insights.
    3. Chain-of-Thought Reasoning: Step-by-step reasoning.
    4. Generated Questions: 3 relevant questions.
    5. Answers: Responses using Neo4j-retrieved context.

