<!-- markdownlint-disable MD033 MD045 -->

# Building RAG with Llama 3.1, Ollama, LlamaIndex & Elasticsearch

<img src="es_rag.gif" style="zoom:50%;" />

## 1 Run Elasticsearch

> Elasticsearch 8.11 is able to automatically map some vector types
>
> Major releases:
>
> 1.0.0 – February 12, 2014
> 2.0.0 – October 28, 2015
> 5.0.0 – October 26, 2016
> 6.0.0 – November 14, 2017
> 7.0.0 – April 10, 2019
> 8.0.0 – February 10, 2022

```sh
export es_version=8.15.0

docker run -d -p 9200:9200 --rm \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.security.http.ssl.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:$es_version
```

## 2 Prepare for Elastic with LlamaIndex

```sh
python -m venv es_venv
# macos | linux
source es_venv/bin/activate
pip install --upgrade pip
# windows
source es_venv/Scripts/activate
python.exe -m pip install --upgrade pip
```

```sh
pip install llama-index llama-index-cli llama-index-core \
  llama-index-embeddings-elasticsearch llama-index-embeddings-ollama \
  llama-index-legacy llama-index-llms-ollama \
  llama-index-readers-elasticsearch llama-index-readers-file \
  llama-index-vector-stores-elasticsearch llamaindex-py-client
```

## 3 Test Ollama

```sh
ollama show llama3.1
```

```sh
curl -X POST http://localhost:11434/api/generate -d '{ "model": "llama3", "prompt":"Why is the sky blue?" }'
```

## 4 Code using LlamaIndex

`hello_es.py`

````python
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

ollama_embedding = OllamaEmbedding("llama3.1")

pipeline = IngestionPipeline(
    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=100), ollama_embedding],
    vector_store=es_vector_store
)
pipeline.run(show_progress=True, documents=documents)

Settings.embed_model = ollama_embedding
local_llm = Ollama(model="llama3.1")
index = VectorStoreIndex.from_vector_store(es_vector_store)
query_engine = index.as_query_engine(local_llm, similarity_top_k=10)

query = "What are the organizations sales goals?"
bundle = QueryBundle(
    query_str=query, embedding=Settings.embed_model.get_query_embedding(query=query))

response = query_engine.query(bundle)

print("response:", response.response)
````

## 5 Hello RAG

```sh
$ python hello_es.py
response: To effectively serve our customers and achieve our business objectives across multiple regions. 

The organization is structured to meet customer needs through collaboration between departments and a focus on delivering high-quality products and services. The teams work closely with marketing, product development, and customer support to ensure consistency in their delivery. This approach enables the company to maintain its position in the market and serve customers effectively.
```

### Elasticsearch Docker & GPU

<img src="../img/res.png" style="zoom:95%;" />

## Reference

- <https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html>
- <https://www.elastic.co/search-labs>
