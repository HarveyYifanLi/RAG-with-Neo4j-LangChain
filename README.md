## RAG-with-Neo4j-LangChain
This is a prototype LangChain system that provides grounding for an LLM using RAG technique backed by Neo4j GraphDB Vector-embedding

![image](https://github.com/HarveyYifanLi/RAG-with-Neo4j-LangChain/assets/17951024/f366ed80-b347-4862-924b-dba72f88f0c5)


### Workflow
  You have a GraphDB (e.g. Neo4J) and want to utilize its valuable data (e.g. relationships between data nodes) to provide grounding to
  an LLM chatbot that your app is connected to. Below is one possible high-level workflow for achieving this.
  
  1. Decide Labels and Properties:
  
  Decide the data Node Labels (i.e. types) and the properties/fields (p.s. typically string fields) of each type of data Nodes for which to create vector embeddings
  
  2. Load the embeddings:
  
  Query the Nodes and then utilize an LLM Provider to convert the targeted property of each queried Node to a vector embedding (i.e. now you'll obtain an additional vector property on this Node)
  
  3. Create the Vector Index:
  
  Create a vector index for the newly created vector embedding property against which you can do similarity search across these new embeddings
  
  4. Create a LangChain retriever from the vector index:
  
  Create a Neo4jVector Retriever from the new vector index
  
  5. Create a LangChain RetrievalQA chain:
  
  Create the RetrievalQA chain using the Neo4jVector retriever
  
  6. Create a LangChain Agent from an LLM chat model:
  
  Create an agent from an LLM chat model and a new LangChain tool to use the RetrievalQA chain, which can then be invoked by issuing queries to it
  
  -> Now the LLM chat model is grounded with realistic data from the GraphDB

### Steps of execution
  1. Install Neo4j Desktop (https://neo4j.com/download/) and load some sample data by running `load-movies.cypher` in the Neo4j browser

  2. Make sure you've signed up for OpenAI (https://openai.com/blog/openai-api) and Comet (https://www.comet.com/signup?utm_source=mit_dl&utm_medium=partner&utm_content=github) and obtained the API keys.
Then put relevant credentials in the `.env` file
   
  3. Clone this repo and cd into its working directory
   
  4. Install, create and activate a Python virtual environment:
     
   -> You can install the virtualvenv Python tool to your host Python by running this command: `pip install virtualenv`
   
   -> Create a virtual environment by running this: `python<version> -m venv <virtual-environment-name>`
   
   -> On a Mac, to activate this virtual environment run: `source <virtual-environment-name>/bin/activate`

  5. Install all the Python dependencies via: `pip install -r requirements.txt`

  6. 

  

  
