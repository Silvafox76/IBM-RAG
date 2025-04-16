from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_core.documents import Document

# Load the new-Policies.txt file
try:
    with open("new-Policies.txt", "r") as f:
        content = f.read()
except FileNotFoundError:
    print("❌ 'new-Policies.txt' not found!")
    exit()

# Wrap content in LangChain Document
doc = [Document(page_content=content)]

# Split the document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = splitter.split_documents(doc)

# Define embedding model
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}
embedder = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)

# Create Chroma vector DB
vectordb = Chroma.from_documents(chunks, embedder)

# Use ChromaDB as a retriever
retriever = vectordb.as_retriever()

# Define query for the retriever
query = "Email policy"
results = retriever.get_relevant_documents(query)

# Display the top 2 results
print("Top 2 Similar Segments for query: Email policy\n")
for i, res in enumerate(results[:2]):
    print(f"--- Result {i+1} ---")
    print(res.page_content)
    print("\n")
