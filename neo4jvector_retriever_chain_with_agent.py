from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_community.tools import YouTubeSearchTool

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_URI = os.getenv('NEO4J_URI')

### build a LLMChain using OpenAI's chat model
openai_chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

promptTemplate = PromptTemplate(
    template="""
    You are an expert in everything. You can explain any topic in details and in a logical fashion.

    Chat History:{chat_history}
    Question:{input}
    """,
    input_variables=["chat_history", "input"],
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_chain = LLMChain(llm=openai_chat_model, prompt=promptTemplate, memory=memory)

### build a RetrievalQA chain from a vector retriever
openai_chat_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
# use OpenAI as the embedding provider because we generated the embedding vector for tagline using OpenAI().embeddings.create()
embedding_provider = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# use Neo4jVector class to create the vector from an existing vector index in Neo4j
# then this vector will be used to create a retriever via movieTaglineIdx_vector.as_retriever()
movieTaglineIdx_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="movieTaglineIdx",
    embedding_node_property="taglineEmbedding",
    text_node_property="tagline",
)

# RetrievalQA chain will use the retriever from movieTaglineIdx_vector to retrieve documents 
# from the movieTaglineIdx index and pass them to the openai_chat_llm language model
retriever_chain = RetrievalQA.from_llm(
    llm=openai_chat_llm,
    retriever=movieTaglineIdx_vector.as_retriever(),
    verbose=True, # by setting the optional verbose and return_source_documents arguments to True when creating the RetrievalQA chain,
    return_source_documents=True # we can see the source documents and the retriever’s score for each document

)

# Each of the Tools expect a single query input and a single output key. The RetrievalQA chain returns multiple output keys.
# As a result, the agent’s tool executor cannot call the RetrievalQA chain directly e.g. using func=retrievalQA.run.
# Thus we wrap the RetrievalQA chain in a function that takes a single string input, format the results and return a single string.
def run_retriever(query):
    results = retriever_chain.invoke({"query": query})
    # format the results
    movies_string = '\n'.join([doc.metadata["title"] + " - " + doc.page_content for doc in results["source_documents"]])
    return movies_string

# tools provide the list of tools that the llm model can access (as per instructions defined in 
# the promptTemplate pulled via hub.pull("whatever-prompt-template"))
# the description field tells the model when to actually use/call the tool
# THUS, a tool does NOT necessarily get triggered. The model will decide whether it will need to call this tool based on user input and its description
youtube = YouTubeSearchTool()

tools = [
    Tool.from_function(
        name="Expertise knowledge Chat",
        description="For when you need to chat about any topic in details or when that topic requires some deep expertise, such as academic knowledge. The question will be a string. Return a string.",
        func=chat_chain.run,
        return_direct=True,
    ),
    Tool.from_function(
        name="Video Search",
        description="Use when needing to find a video on any topic such as a movie trailer. The question will likely include the word 'video'. Return a link to a YouTube video.",
        func=youtube.run,
        return_direct=True,
    ),
    Tool.from_function(
        name="Movie Tagline Search",
        description="For when you need to compare a tagline to a movie. The question will be a string. Return a string.",
        func=run_retriever,
        return_direct=True
    )
]

### build the langchain agent that can use the set of tools that we defined
# hub.pull("hwchase17/react-chat") defines a customized promptTemplate similar to our general promptTemplate
agent_prompt = hub.pull("hwchase17/react-chat")
# instantiation of an agent requires a LLM model, a list of tools and a default promptModel for the agent
agent = create_react_agent(openai_chat_model, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    max_interations=3,
    verbose=True,
    handle_parse_errors=True,
)

while True:
    q = input("> ")
    response = agent_executor.invoke({"input": q})
    print(response["output"])
