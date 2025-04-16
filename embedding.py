from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

# Set embedding parameters
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

# Initialize the Watsonx embedding model
embedder = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)

# Define the query
query = "How are you?"

# Generate the embedding
embedding = embedder.embed_query(query)

# Print the first 5 embedding values
print("First 5 embedding values:")
print(embedding[:5])
