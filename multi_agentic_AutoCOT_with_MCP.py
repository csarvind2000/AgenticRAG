import gradio as gr
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from typing import TypedDict, List, Dict
import PyPDF2
import re
import io

# Define the state with MCP-inspired context
class AgentState(TypedDict):
    context: Dict[str, str]  # MCP: Metadata like source, stage
    input_data: str
    cleaned_data: str
    analyzed_data: str
    cot_reasoning: str
    questions: List[str]
    answers: Dict[str, str]
    documents: List[str]  # For RAG retrieval

# Initialize the LLM with Ollama (LLaMA 3)
llm = Ollama(model="llama3", temperature=0.7)

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Agent 0: Text Cleaner
text_cleaner_prompt = PromptTemplate(
    input_variables=["input_data", "context"],
    template="Context: {context}\n\nAnalyze the following text and identify repetitive patterns or noise. Provide a list of Python `re.sub` commands to remove these patterns, focusing on excessive repetitions or irrelevant sequences. Return as a numbered list:\n\n{input_data}"
)
text_cleaner_chain = text_cleaner_prompt | llm | StrOutputParser()

def text_cleaner_node(state: AgentState) -> AgentState:
    state["context"]["stage"] = "text_cleaning"
    cleaning_suggestions = text_cleaner_chain.invoke({
        "input_data": state["input_data"],
        "context": str(state["context"])
    })
    
    cleaned_text = state["input_data"]
    for line in cleaning_suggestions.split("\n"):
        if line.strip().startswith("1.") or line.strip().startswith("-") or "re.sub" in line:
            try:
                match = re.search(r"re\.sub\((r?['\"].*?['\"]),\s*(r?['\"].*?['\"]),", line)
                if match:
                    pattern, replacement = match.groups()
                    pattern = pattern.strip("'\"")
                    replacement = replacement.strip("'\"")
                    cleaned_text = re.sub(pattern, replacement, cleaned_text)
            except Exception as e:
                print(f"Error parsing re.sub from line '{line}': {e}")
    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    state["cleaned_data"] = cleaned_text
    state["context"]["cleaned_length"] = str(len(cleaned_text))
    return state

# Agent 1: Data Analyzer
data_analyzer_prompt = PromptTemplate(
    input_variables=["cleaned_data", "context"],
    template="Context: {context}\n\nAnalyze the following cleaned text and extract key insights in a concise summary:\n\n{cleaned_data}"
)
data_analyzer_chain = data_analyzer_prompt | llm | StrOutputParser()

def data_analyzer_node(state: AgentState) -> AgentState:
    state["context"]["stage"] = "data_analysis"
    analyzed_data = data_analyzer_chain.invoke({
        "cleaned_data": state["cleaned_data"],
        "context": str(state["context"])
    })
    state["analyzed_data"] = analyzed_data
    state["documents"] = [state["cleaned_data"][i:i+100] for i in range(0, len(state["cleaned_data"]), 100)]
    state["context"]["analyzed_at"] = "April 08, 2025"  # Example metadata
    return state

# Agent 2: CoT Builder
cot_builder_prompt = PromptTemplate(
    input_variables=["analyzed_data", "context"],
    template="Context: {context}\n\nBased on this summary, construct a step-by-step Chain-of-Thought reasoning:\n\n{analyzed_data}"
)
cot_builder_chain = cot_builder_prompt | llm | StrOutputParser()

def cot_builder_node(state: AgentState) -> AgentState:
    state["context"]["stage"] = "cot_building"
    cot_reasoning = cot_builder_chain.invoke({
        "analyzed_data": state["analyzed_data"],
        "context": str(state["context"])
    })
    state["cot_reasoning"] = cot_reasoning
    return state

# Agent 3: Question Generator
question_generator_prompt = PromptTemplate(
    input_variables=["cot_reasoning", "analyzed_data", "context"],
    template="Context: {context}\n\nBased on the reasoning and summary below, generate 3 relevant questions:\n\nReasoning:\n{cot_reasoning}\n\nSummary:\n{analyzed_data}"
)
question_generator_chain = question_generator_prompt | llm | StrOutputParser()

def question_generator_node(state: AgentState) -> AgentState:
    state["context"]["stage"] = "question_generation"
    questions_text = question_generator_chain.invoke({
        "cot_reasoning": state["cot_reasoning"],
        "analyzed_data": state["analyzed_data"],
        "context": str(state["context"])
    })
    state["questions"] = [q.strip() for q in questions_text.split("\n") if q.strip()][:3]
    return state

# Agent 4: Answer Provider (with RAG)
answer_provider_prompt = PromptTemplate(
    input_variables=["question", "context", "cot_reasoning", "rag_context"],
    template="Context: {context}\n\nUsing the RAG context and reasoning below, answer the question:\n\nQuestion: {question}\n\nRAG Context: {rag_context}\n\nReasoning: {cot_reasoning}"
)
answer_provider_chain = answer_provider_prompt | llm | StrOutputParser()

def answer_provider_node(state: AgentState) -> AgentState:
    state["context"]["stage"] = "answer_providing"
    vector_store = FAISS.from_texts(state["documents"], embeddings)
    answers = {}
    for question in state["questions"]:
        docs = vector_store.similarity_search(question, k=2)
        rag_context = "\n".join([doc.page_content for doc in docs])
        answer = answer_provider_chain.invoke({
            "question": question,
            "context": str(state["context"]),
            "cot_reasoning": state["cot_reasoning"],
            "rag_context": rag_context
        })
        answers[question] = answer
    state["answers"] = answers
    return state

# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("text_cleaner", text_cleaner_node)
workflow.add_node("data_analyzer", data_analyzer_node)
workflow.add_node("cot_builder", cot_builder_node)
workflow.add_node("question_generator", question_generator_node)
workflow.add_node("answer_provider", answer_provider_node)
workflow.add_edge("text_cleaner", "data_analyzer")
workflow.add_edge("data_analyzer", "cot_builder")
workflow.add_edge("cot_builder", "question_generator")
workflow.add_edge("question_generator", "answer_provider")
workflow.add_edge("answer_provider", END)
workflow.set_entry_point("text_cleaner")
graph = workflow.compile()

# Gradio processing function
def process_brochure(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF file."
    
    pdf_content = extract_text_from_pdf(pdf_file)
    if not pdf_content.strip():
        return "No text could be extracted from the PDF."
    
    initial_state = {
        "context": {"source": "PDF", "stage": "initial"},
        "input_data": pdf_content,
        "cleaned_data": "",
        "questions": [],
        "answers": {},
        "documents": []
    }
    result = graph.invoke(initial_state)
    
    output = f"### Context\n{result['context']}\n\n"
    output += f"### Cleaned Data\n{result['cleaned_data']}\n\n"
    output += f"### Analyzed Data\n{result['analyzed_data']}\n\n"
    output += f"### Chain-of-Thought Reasoning\n{result['cot_reasoning']}\n\n"
    output += "### Generated Questions\n"
    for q in result['questions']:
        output += f"- {q}\n"
    output += "\n### Answers\n"
    for q, a in result['answers'].items():
        output += f"**Q: {q}**\nA: {a}\n\n"
    
    return output

# Gradio UI
with gr.Blocks(title="Carnival Brochure Analyzer with MCP") as demo:
    gr.Markdown("# Carnival Brochure Analyzer with MCP")
    gr.Markdown("Upload a PDF brochure to analyze it with a multi-agent RAG system using a Model Context Protocol.")
    
    pdf_input = gr.File(label="Upload PDF Brochure", file_types=[".pdf"])
    output_box = gr.Markdown(label="Analysis Results")
    submit_btn = gr.Button("Analyze")
    
    submit_btn.click(
        fn=process_brochure,
        inputs=pdf_input,
        outputs=output_box
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()