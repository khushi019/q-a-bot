# Loan Prediction Q&A System

[![Python Application](https://github.com/khushi019/q-a-bot/actions/workflows/python-app.yml/badge.svg)](https://github.com/khushi019/q-a-bot/actions/workflows/python-app.yml)

A Retrieval-Augmented Generation (RAG) based question-answering system for loan prediction data.

## Features

- Interactive Q&A interface using Streamlit
- Semantic search to retrieve relevant loan records
- AI-powered answers based on retrieved context
- Detailed source information
- Dataset statistics


## Setup

1. Clone this repository
   ```
   git clone https://github.com/khushi019/q-a-bot.git
   cd q-a-bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your HuggingFace API token:
   ```
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
   
   # Optional: Change the default model
   # HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
   ```
   Get your token at: https://huggingface.co/settings/tokens
   
   **Note:** The application uses `mistralai/Mistral-7B-Instruct-v0.2` by default, which is an open-source model available on HuggingFace.

## Usage

1. Run the application:
   ```
   streamlit run app.py
   ```
2. Open your browser at the URL shown in the terminal (typically http://localhost:8501)
3. Enter your questions about loan applications in the text input
4. Adjust the number of documents to retrieve and search method as needed
5. Click "Get Answer" to generate a response

## Example Questions

- What factors affect loan approval?
- How does credit history impact loan status?
- What percentage of loans are approved for self-employed applicants?
- Is there a correlation between income and loan approval?
- What is the average loan amount for approved applications?

## System Architecture

This system uses:
- Streamlit for the web interface
- LangChain for document processing and retrieval
- HuggingFace for embeddings and LLM inference
- ChromaDB for vector storage

## Data

The system uses the "Training Dataset.csv" file which contains loan application data with features like:
- Loan ID
- Gender
- Marital status
- Dependents
- Education
- Self-employment status
- Income (applicant and co-applicant)
- Loan amount and term
- Credit history
- Property area
- Loan status (approved/rejected)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: [Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- Built with [LangChain](https://github.com/langchain-ai/langchain) and [Streamlit](https://streamlit.io/)
