A prototype LangChain system that provides grounding for an LLM using RAG technique backed by Neo4j GraphDB Vector-embedding

![image](https://github.com/HarveyYifanLi/RAG-with-Neo4j-LangChain/assets/17951024/f366ed80-b347-4862-924b-dba72f88f0c5)


- Workflow
  
  You have a GraphDB (e.g. Neo4J) and want to utilize its valuable data (e.g. relationships between data nodes) to provide grounding to
  an LLM chatbot that your app is connected to. Below is one possible high-level workflow.
  
  -> Decide Labels and Properties:
  
  Decide the data Node Labels (i.e. types) and the properties/fields (p.s. typically string fields) of each type of data Nodes for which to create vector embeddings
  
  -> Load the embeddings:
  
  Query the Nodes and then utilize an LLM Provider to convert the targeted property of each queried Node to a vector embedding (i.e. now you'll obtain an additional vector property on this Node)
  
  -> Create the Vector Index:
  
  Create a vector index for the newly created vector embedding property against which you can do similarity search across these new embeddings
  
  -> Create a LangChain retriever from the vector index:
  
  Create a Neo4jVector Retriever from the new vector index
  
  -> Create a LangChain RetrievalQA chain:
  
  Create the RetrievalQA chain using the Neo4jVector retriever
  
  -> Create a LangChain Agent from an LLM chat model:
  
  Create an agent from an LLM chat model and a new LangChain tool to use the RetrievalQA chain
  
  -> Now the LLM chat model is grounded with the realistic data from the GraphDB data

- Steps of execution
  
    1. 
    
