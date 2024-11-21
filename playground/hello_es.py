from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.llms.ollama import Ollama
from llama_index.core import Document, Settings
# from urllib.request import urlopen
import json

es_vector_store = ElasticsearchStore(
    es_url="http://127.0.0.1:9200",
    index_name="workplace_index",
    vector_field='content_vector',
    text_field='content'
)

# url = "https://raw.githubusercontent.com/elastic/elasticsearch-labs/main/datasets/workplace-documents.json"
# response = urlopen(url)
# workplace_docs = json.loads(response.read())
with open('workplace-documents.json', 'r') as file:
    workplace_docs = json.load(file)

documents = [
    Document(
        text=doc['content'],
        metadata={"name": doc['name'], "summary": doc['summary'],
                  "rolePermissions": doc['rolePermissions']}
    ) for doc in workplace_docs
]

# Embedding Model to do local embedding using Ollama.
ollama_embedding = OllamaEmbedding("llama3.1")

#
# LlamaIndex Pipeline configured to take care of chunking, embedding
# and storing the embeddings in the vector store.
# pipeline = IngestionPipeline(
#     transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=100), ollama_embedding],
#     vector_store=es_vector_store
# )
# pipeline.run(show_progress=True, documents=documents)
#

Settings.embed_model = ollama_embedding
local_llm = Ollama(model="llama3.1")
index = VectorStoreIndex.from_vector_store(es_vector_store)
query_engine = index.as_query_engine(local_llm, similarity_top_k=10)

# Customer Query
query = "What are the organizations sales goals?"
bundle = QueryBundle(
    query_str=query, embedding=Settings.embed_model.get_query_embedding(query=query))

response = query_engine.query(bundle)

print("response:", response.response)
