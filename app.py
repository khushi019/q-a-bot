import os
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv

from huggingface_hub import InferenceClient

# Update LangChain imports to match the current package structure
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Loan Prediction Q&A",
    page_icon="💰",
    layout="wide"
)

# Initialize HuggingFace client
@st.cache_resource
def get_client():
    try:
        # Get model from environment variable or use default
        # Using a model that's more widely available and works with the HuggingFace API
        model_name = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        return InferenceClient(
            model=model_name,
            token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
    except Exception as e:
        st.error(f"Error initializing HuggingFace client: {e}")
        return None

client = get_client()

def generate(query_context: str) -> str:
    """Generate response using the LLM."""
    try:
        if client is None:
            return "Error: HuggingFace client not initialized. Please check your API token."
        
        try:
            # First try chat completion
            resp = client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about loan applications based on provided context."},
                    {"role": "user", "content": query_context}
                ],
                max_tokens=512,
                temperature=0.7
            )
            
            # Parse chat completion response
            if hasattr(resp, 'choices') and len(resp.choices) > 0:
                return resp.choices[0].message.content
            elif isinstance(resp, dict) and "choices" in resp and len(resp["choices"]) > 0:
                return resp["choices"][0]["message"]["content"]
            else:
                return str(resp)
                
        except Exception as chat_error:
            # If chat completion fails, fall back to text generation
            st.warning(f"Chat completion failed, falling back to text generation: {chat_error}")
            
            # Format prompt for text generation
            prompt = f"""<s>[INST] You are a helpful assistant that answers questions about loan applications based on provided context.

Context information:
{query_context}

Please provide a detailed and helpful answer based on the context. [/INST]</s>
"""
            
            resp = client.text_generation(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            
            # Handle text generation response
            if isinstance(resp, str):
                return resp
            elif hasattr(resp, 'generated_text'):
                return resp.generated_text
            elif isinstance(resp, dict) and "generated_text" in resp:
                return resp["generated_text"]
            elif isinstance(resp, list) and len(resp) > 0:
                return resp[0].get("generated_text", str(resp[0]))
            else:
                return str(resp)
            
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response. Please try again."

# Load and process the dataset
@st.cache_data
def load_docs():
    """Load and process loan dataset into documents."""
    try:
        df = pd.read_csv("data/Training Dataset.csv")
        
        # Handle numeric and non-numeric columns separately to avoid dtype warning
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
        
        # Fill numeric columns with -1 or 0 instead of "Unknown"
        df[numeric_cols] = df[numeric_cols].fillna(-1)
        df[non_numeric_cols] = df[non_numeric_cols].fillna("Unknown")
        
        docs = []
        for idx, row in df.iterrows():
            # Create a more detailed document for each loan application
            content = (
                f"Loan ID: {row.Loan_ID}\n"
                f"Applicant Details: Gender: {row.Gender}, Married: {row.Married}, "
                f"Dependents: {row.Dependents}, Education: {row.Education}, "
                f"Self-Employed: {row.Self_Employed}\n"
                f"Financial Information: Applicant Income: ${row.ApplicantIncome}, "
                f"Co-applicant Income: ${row.CoapplicantIncome}, "
                f"Loan Amount: ${row.LoanAmount}, Loan Term: {row.Loan_Amount_Term} months\n"
                f"Credit History: {row.Credit_History}, Property Area: {row.Property_Area}\n"
                f"Loan Status: {'Approved' if row.Loan_Status == 'Y' else 'Rejected'}"
            )
            docs.append(Document(page_content=content, metadata={"id": row.Loan_ID, "index": idx}))
        
        return docs
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return []

# Build vectorstore for semantic search
@st.cache_resource
def get_vectorstore(_docs):
    """Create and return a vector store from documents."""
    try:
        if not _docs:
            return None
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(_docs)
        
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma.from_documents(chunks, embed, persist_directory="emb_store")
    except Exception as e:
        st.error(f"Error building vector store: {e}")
        return None

# Create a better prompt template
def create_prompt(context: str, query: str) -> str:
    """Create a well-structured prompt for the LLM."""
    return f"""You are a helpful assistant that answers questions about loan applications based on the provided context.
    
Context information:
{context}

Question: {query}

Provide a detailed and helpful answer based only on the information in the context. If the information needed is not in the context, say "I don't have enough information to answer this question."

Answer:"""

# Main UI
st.title("💰 Loan Prediction Q&A System")
st.markdown("Ask questions about loan applications and get AI-powered answers based on the training dataset.")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.info("""
    This application uses RAG (Retrieval-Augmented Generation) to answer questions about loan applications.
    
    It retrieves relevant loan records from the dataset and uses an LLM to generate answers based on the retrieved context.
    """)
    
    # Display model information
    st.header("Model Information")
    model_name = os.getenv("HUGGINGFACE_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    st.write(f"Using model: **{model_name}**")
    if client is None:
        st.error("⚠️ HuggingFace client not initialized")
    else:
        st.success("✅ HuggingFace client initialized")
    
    st.header("Dataset Statistics")
    try:
        df = pd.read_csv("data/Training Dataset.csv")
        st.write(f"Total records: {len(df)}")
        st.write(f"Approved loans: {len(df[df['Loan_Status'] == 'Y'])}")
        st.write(f"Rejected loans: {len(df[df['Loan_Status'] == 'N'])}")
    except:
        st.write("Could not load dataset statistics")

# Load documents and create vector store
docs = load_docs()
vs = get_vectorstore(docs)

# Main query interface
query = st.text_input("Ask a question about loan applications:", placeholder="E.g., What factors affect loan approval?")
col1, col2 = st.columns([1, 3])

with col1:
    k_value = st.slider("Number of documents to retrieve:", min_value=1, max_value=10, value=3)
    
with col2:
    search_type = st.radio("Search method:", ["Semantic Search", "Keyword Search"], horizontal=True)

if st.button("Get Answer") and query and vs:
    # Check if client is initialized
    if client is None:
        st.error("HuggingFace client not initialized. Please check your API token.")
    else:
        with st.spinner("Searching and generating answer..."):
            try:
                # Retrieve relevant documents
                if search_type == "Semantic Search":
                    dlist = vs.similarity_search(query, k=k_value)
                else:
                    dlist = vs.max_marginal_relevance_search(query, k=k_value, fetch_k=k_value*3)
                
                # Create context from retrieved documents
                ctx = "\n\n---\n\n".join([d.page_content for d in dlist])
                
                # Generate the prompt and get answer
                prompt = create_prompt(ctx, query)
                answer = generate(prompt)
                
                # Display results
                st.subheader("Answer")
                st.write(answer)
                
                # Display sources with expanders
                st.subheader("Sources")
                for i, doc in enumerate(dlist):
                    with st.expander(f"Source {i+1}"):
                        st.write(doc.page_content)
            except Exception as e:
                st.error(f"Error processing query: {e}")
else:
    if not vs and query:
        st.error("Vector store could not be initialized. Please check the logs.")
