## RAG-with-Neo4j-LangChain
This is a prototype LangChain system that provides grounding for an LLM using RAG technique backed by Neo4j GraphDB Vector-embedding

![image](https://github.com/HarveyYifanLi/RAG-with-Neo4j-LangChain/assets/17951024/f366ed80-b347-4862-924b-dba72f88f0c5)

### Background
  Retrieval-Augmented Generation (RAG) is the process/technique of optimizing the output of a large language model by referencing an authoritative knowledge base outside of its 
  training data before generating a response which as a result overcomes the so-called LLM Hallucination problem. This process is also more generally known as grounding an LLM.
  This app focuses on grounding an LLM with the authoritative data source being a Graph Database such as Neo4j; therefore, it assumes you already have a GraphDB available for connection. It uses a sample Movie dataset as an example and so please feel free to update the data Node's Label and Property names to cater to your use case.

### Workflow
  You have a GraphDB (e.g. Neo4j) and want to utilize its valuable data (e.g. relationships between data nodes) to provide grounding to
  an LLM chatbot that your app is connected to. Below is one possible high-level workflow for achieving this.
  
  1. Decide Labels and Properties:
  
  Decide the data Node Labels (i.e. types) and the properties/fields (p.s. typically string fields) of each type of data Nodes for which to create vector embeddings
  
  2. Load the embeddings:
  
  Query the Nodes and then utilize an LLM Provider to convert the targeted property of each queried Node to a vector embedding (i.e. now you'll obtain an additional vector property on this Node)
  
  3. Create the Vector Index:
  
  Create a vector index for the newly created vector embedding property against which you can do similarity search across these new embeddings
  
  4. Create a LangChain retriever from the vector index:
  
  Create a `Neo4jVector` Retriever from the new vector index
  
  5. Create a LangChain RetrievalQA chain:
  
  Create the `RetrievalQA` chain using the `Neo4jVector` retriever
  
  6. Create a LangChain Agent from an LLM chat model:
  
  Create an `agent` from an LLM chat model and a new LangChain tool to use the `RetrievalQA` chain, which can then be invoked by issuing queries to it
  
  -> Now the LLM chat model is grounded with realistic data from the GraphDB

### Steps of execution
  1. Install [Neo4j Desktop](https://neo4j.com/download/) and load some sample data by running `load-movies.cypher` in the Neo4j browser

  2. Make sure you've signed up for [OpenAI](https://openai.com/blog/openai-api) and [Comet](https://www.comet.com/signup?utm_source=mit_dl&utm_medium=partner&utm_content=github) and obtained the API keys.
Then put relevant credentials in the `.env` file (by modifying the `.env.example` file)
   
  3. Clone this repo and cd into its working directory
   
  4. Install, create and activate a Python virtual environment:
     
   - You can install the virtualvenv Python tool to your host Python by running this command: `pip install virtualenv`
   
   - Create a virtual environment by running this: `python<version> -m venv <virtual-environment-name>`
   
   - On a Mac, to activate this virtual environment run: `source <virtual-environment-name>/bin/activate`

  5. Install all the Python dependencies via: `pip install -r requirements.txt`

  6. Generate vector embeddings and create a vector index for it in Neo4j using OpenAI:
     
   - Run command: `python<version> openai_embeddings.py` and it will:
     - use OpenAI to generate a vector embedding for the `tagline` property (if it exists) of `movie` Nodes in Neo4j then save them as rows in a csv file, which could be loaded into Neo4j for further processing
       
       ```
       LLM providers typically expose API endpoints that convert a chunk of text into a vector embedding.
       Depending on the provider, the shape and size of the vector may differ.
       For example, OpenAI’s text-embedding-ada-002 embedding model converts text into a vector of 1,536 dimensions.
        ```

     - load OpenAI vector embedding as a new vector property `taglineEmbedding` onto the `movie` Node in Neo4j (by reading/loading the csv file)
     - create a vector index `movieTaglineIdx` in Neo4j for this new vector property that's used for the embedding, so as to do search across these embeddings
     - (and finally also do a test query using the created vector index to find the closest embedding matches to a given embedding)

  7. Create an LLMChain, a RetrievalQA Chain from the Neo4jVector retriever based on the vector index and an agent to use the RetrievalQA chain as a tool:

   - Run command: `python<version> neo4jvector_retriever_chain_with_agent.py` and it will:
      - create and use an `LLMChain` from an OpenAI chat model that supports a Conversation Buffer memory
      - Create a `Neo4jVector` Retriever from the new vector index
      - Create the `RetrievalQA` chain using the `Neo4jVector` retriever
      - Create an `agent` from the LLM chat model and a new LangChain tool to use the `RetrievalQA` chain, which can then be invoked by issuing queries to it

  8. Now feel free to interact with the grounded LLM:

      ![image](https://github.com/HarveyYifanLi/RAG-with-Neo4j-LangChain/assets/17951024/eacede0f-1e47-44f0-b57a-42d1127359a5)


  
