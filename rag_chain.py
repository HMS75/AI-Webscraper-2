from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama  # Use langchain_ollama if you're using latest versions

def build_rag_chain(vectordb):
    # Step 1: Set up retriever from vector DB
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Step 2: Set up LLM (Gemma via Ollama) with temperature=0
    llm = Ollama(model="gemma3:1b", temperature=0)

    # Step 3: Create a clear custom prompt
    prompt_template = """
You are a helpful assistant. Use the following context to answer the question accurately.

📄 Context:
{context}

❓ Question:
{question}

🧠 Answer:
""".strip()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Step 4: Build a QA chain using the prompt
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

    # Step 5: Build final RAG pipeline
    rag_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=qa_chain,
        return_source_documents=True  # Optional: for debugging or display
    )

    return rag_chain
