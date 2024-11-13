# Wiki Bot🤖: Your Wikipedia Companion

## Overview
The objective of this project is to build a Wikipedia-based chatbot, WikiBot🤖, using Retrieval-Augmented Generation (RAG). The chatbot leverages LangChain, HuggingFace models, and a Chroma vector store to answer user queries based on real-time retrieval of Wikipedia articles. By combining large language models with retrieval techniques, WikiBot provides contextually relevant and accurate answers to user queries, effectively utilizing Wikipedia data.

The project involves:

- **Streamlit**: Delivers an interactive user interface for real-time user interaction.
- **requests and Beautiful Soup libs**: requests fetches the HTML content of the specific **Wikipedia** page, while Beautiful Soup extracts main content from it.
- **RecursiveTextSplitter**: Splits the text content scrapped from **Wikipedia** into manageable chunks.
- **Hugging Face's sentence-transformers/all-MiniLM-L6-v2 Model**: Creates text embeddings.
- **Chroma Vector Database**: Efficiently stores and retrieves text embeddings for semantic search.
- **LangChain**: Integrates the language model (LLM) with the vector database.
- **Meta-Llama-3-8B-Instruct with HuggingFace API**: Provides the processing power for natural language understanding.
- **Directive-based prompt pattern**: Guides the language model on how to generate appropriate responses based on the query context and user interaction directives.

## Architecture Overview
![Wikibot](https://github.com/user-attachments/assets/af0d552a-b983-4198-9e0f-2e71718afc86)

## Instructions on How to Setup and Run

### Step 1: Install Required Python Libraries

Install the necessary libraries from the requirements.txt file:

```bash
pip install -r requirements.txt
```
**NOTE:**
If there is issue with Chroma and Pysqlite3 installation, follow below steps 
- Download Python 3.11 version or above and then install chromadb using below command
```bash
pip install chromadb
``` 
- For pysqlite3 installation keep the wheel file (available in Github repo) in your project directory and run below command
```bash
pip install pysqlite3_wheels-0.5.0-cp311-cp311-win_amd64
```

### Step 2: Launch Wiki_Bot

1. **Set Up API Key**: Ensure your HuggingFace API Token is in the .env file
```bash
HUGGINGFACEHUB_API_TOKEN = "<HUGGINGFACEHUB_API_TOKEN>"
```
2. **Launch**: After setting up the API Key, launch the Wiki Bot interface by running:
```bash
streamlit run FashionBot.py
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any queries or contributions, feel free to reach out to:
- **Ariharasudhan A** - [Email](mailto:ariadaikalam1234@gmail.com)
