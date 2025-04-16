from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_core.documents import Document

# Step 1: Load the text file
try:
    with open("new-Policies.txt", "r") as f:
        content = f.read()
except FileNotFoundError:
    print("❌ File 'new-Policies.txt' not found!")
    exit()

# Step 2: Wrap it in a LangChain Document
doc = [Document(page_content=content)]

# Step 3: Split the document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = splitter.split_documents(doc)

# Step 4: Initialize embedding model
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

# Step 5: Create Chroma vector DB
vectordb = Chroma.from_documents(chunks, embedder)

# Step 6: Similarity search
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_core.documents import Document

# Step 1: Load the text file
try:
    with open("new-Policies.txt", "r") as f:
        content = f.read()
except FileNotFoundError:
    print("❌ File 'new-Policies.txt' not found!")
    exit()

# Step 2: Wrap it in a LangChain Document
doc = [Document(page_content=content)]

# Step 3: Split the document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = splitter.split_documents(doc)

# Step 4: Initialize embedding model
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

# Step 5: Create Chroma vector DB
vectordb = Chroma.from_documents(chunks, embedder)

# Step 6: Similarity search
query = "Smoking policy"
results = vectordb.similarity_search(query, k=5)

# Step 7: Output the results
print("Top 5 Similar Results for query: Smoking policy\n")
for i, res in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(res.page_content)
    print("\n")
