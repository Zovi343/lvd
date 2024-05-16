# LVD: Learned Vector Database
LVD is a vector database that allows you to store and query embeddings. It is built on top of fork of ChromaDB. 
Internally, LVD uses learned indexing for management of unstructured data. As of time of this writting, LVS is only
database that uses such indexing.

## Setup
To start using this project first make sure that LMI index git submodule is initialized.
```bash
git submodule update --init --recursive
```

Create python conda environment and activate it.
```bash
conda create --name lvd_env python=3.8.18
conda activate lvd_env
```

Next install the dependencies.
```bash
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

Install the torch library. The library is used for training the LMI index.
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Create kernel for the Jupyter notebook.
```bash
python -m ipykernel install --user --name lvd_env --display-name "lvd_env"
```

## Demo

### System Usage
Next run the command bellow to open `system_usage.ipynb`.
```bash
jupyter notebook ./lvd_notebooks/system_usage.ipynb --notebook-dir=./ 
```
After the notebook opens do not forget to change the kernel in the Jupyter interface to `lvd_env` that was created in the previous step.
This notebook demonstrates usage of the LVD system  for management of unstructured data. All the operations supported by LVD are used in this notebook, an example of
some of the data retrieval operations:
```python
# Constrained Search Operation
results = collection.query(
    query_embeddings=[[1, 2, 3]],
    include=["metadatas", 'documents', 'distances'],
    where={"cluster": "red"},
    n_results=5,
    n_buckets=1,
)

# Hybrid Search Operation
results = collection.query(
    query_embeddings=[[1, 2, 3]],
    include=["metadatas", 'distances', "documents"],
    n_results=5,
    n_buckets=1,
    where_document={"$hybrid":{ "$hybrid_terms": ["digital", "data", "programming"]}}
)
```

### RAG Usage
Next run the command bellow to open `rag_usage.ipynb`.
```bash
jupyter notebook ./lvd_notebooks/rag_usage.ipynb --notebook-dir=./ 
```
The notebook `rag_usage.ipynb` shows how the LVD can be used within the RAG architecture.
For this demonstration the LVD is combined with OpenAI ChatGPT 3.5. You can use any other LLM model if you want.
The Arxiv dataset that contains scientific papers is used in this demonstration. 
The documents from the dataset have already been pre-split (pre-chunked) by the authors of the dataset.
The LVD is used to store the chunked documents and serve them as a context to the LLM model.

## Server
You can also set up LVD server locally or deploy it to Kubernetes cluster. Then you can use the LVD client to interact with the server.

### Docker
The LVD is dockerized and can run locally in docker container or can be deployed in Kubernetes cluster.
The `Dockerfile` in the root directory defines the docker image. 
The `deployment.yaml` file in the root directory then uses this image to deploy the LVD server in Kubernetes cluster.

To build Docker image run following command.
```bash
docker build -t lvd .
```

Start the LVD server through Dokcer container and listen on port 5000.
```bash
docker run -p 5000:8000 lvd
```

### Client
If you want to use LVD client outside of this repository you can install it as a package from this repository into your Python environment.
This package simply wraps and modifies original ChromaDB client to work with LVD.

Install LVD package from this repository.
```bash 
python setup.py sdist
pip install ./dist/lvd-0.1.tar.gz
```

Import client from the package to interact with the LVD instance. The package has still same name as ChromaDB. 
Therefore, it can not be installed together with ChromaDB. Since that causes name conflicts.
```python
from chromadb import HttpClient
```

# ChromaDB Inherited README
The rest of the documentation is from the original ChromaDB repository. It describes additional features of the database.
Since LVD is based on the ChromaDB it inherits all of these features.

## Features
- __Simple__: Fully-typed, fully-tested, fully-documented == happiness
- __Integrations__: [`🦜️🔗 LangChain`](https://blog.langchain.dev/langchain-chroma/) (python and js), [`🦙 LlamaIndex`](https://twitter.com/atroyn/status/1628557389762007040) and more soon
- __Dev, Test, Prod__: the same API that runs in your python notebook, scales to your cluster
- __Feature-rich__: Queries, filtering, density estimation and more
- __Free & Open Source__: Apache 2.0 Licensed

## Use case: ChatGPT for ______

For example, the `"Chat your data"` use case:
1. Add documents to your database. You can pass in your own embeddings, embedding function, or let Chroma embed them for you.
2. Query relevant documents with natural language.
3. Compose documents into the context window of an LLM like `GPT3` for additional summarization or analysis.

## Embeddings?

What are embeddings?

- [Read the guide from OpenAI](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)
- __Literal__: Embedding something turns it from image/text/audio into a list of numbers. 🖼️ or 📄 => `[1.2, 2.1, ....]`. This process makes documents "understandable" to a machine learning model.
- __By analogy__: An embedding represents the essence of a document. This enables documents and queries with the same essence to be "near" each other and therefore easy to find.
- __Technical__: An embedding is the latent-space position of a document at a layer of a deep neural network. For models trained specifically to embed data, this is the last layer.
- __A small example__: If you search your photos for "famous bridge in San Francisco". By embedding this query and comparing it to the embeddings of your photos and their metadata - it should return photos of the Golden Gate Bridge.

Embeddings databases (also known as **vector databases**) store embeddings and allow you to search by nearest neighbors rather than by substrings like a traditional database. By default, Chroma uses [Sentence Transformers](https://docs.trychroma.com/embeddings#sentence-transformers) to embed for you but you can also use OpenAI embeddings, Cohere (multilingual) embeddings, or your own.

## License

[Apache 2.0](./LICENSE)
