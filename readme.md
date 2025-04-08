# Multi-agent Retrieval-Augmented Generation (RAG) with Neo4j Knowledge Graph

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

### Cleaned Data
The Power to Surprise With an infinitely adaptable interior and easy access from every angle, the Carnival puts you in total control. It can take a large group of family or friends as far as the roads can reach, with ample space for adventures, tools, or furniture. From futuristic headlamps in the tiger-nose grille to dramatic rear tail lamps, the design is luxurious. Interior is only available in black + grey for Singapore. The quieter diesel engine offers significant fuel savings with improved cooling and reduced friction.

### Analyzed Data
The Kia Carnival features a versatile and spacious interior for up to 8 passengers, a luxurious design with LED lighting and a tiger-nose grille, and a fuel-efficient diesel engine with an 8-speed transmission. The interior, exclusive to black + grey in Singapore, emphasizes comfort and adaptability.

### Chain-of-Thought Reasoning
1. The interior’s adaptability allows it to serve multiple purposes, from passenger transport to cargo storage.
2. The design, with LED lights and a tiger-nose grille, positions it as a premium vehicle.
3. The diesel engine’s efficiency stems from improved cooling and reduced friction, enhancing fuel savings.

### Generated Questions
- How does the diesel engine contribute to fuel efficiency?
- What design elements make the Carnival luxurious?
- Why is the interior color limited to black + grey in Singapore?

### Answers
**Q: How does the diesel engine contribute to fuel efficiency?**
A: The diesel engine improves fuel efficiency through enhanced cooling and reduced friction, as noted in the brochure, paired with an 8-speed transmission for smoother gear changes.

**Q: What design elements make the Carnival luxurious?**
A: The Carnival’s luxurious design includes futuristic headlamps embedded in the tiger-nose grille and dramatic rear tail lamps, creating a premium aesthetic.

