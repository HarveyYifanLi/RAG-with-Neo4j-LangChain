import os
import csv

from openai import OpenAI
from neo4j import GraphDatabase

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_movie_plots(limit=None):
    """
    connect to Neo4j DB and return movies matching the query
    """
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )

    driver.verify_connectivity()
    # Note on using the ID() function to obtain the id of each node
    query = """MATCH (m:Movie) WHERE m.tagline IS NOT NULL
    RETURN ID(m) AS movieId, m.title AS title, m.tagline AS tagline"""

    if limit is not None:
        query += f' LIMIT {limit}'

    movies, summary, keys = driver.execute_query(
        query
    )

    # print((movies, summary, keys))
    driver.close()

    return movies


def generate_embeddings(file_name, limit=None):
    csvfile_out = open(file_name, 'w', encoding='utf8', newline='')
    fieldnames = ['movieId','embedding']

    output_plot = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
    output_plot.writeheader()
    # get all the matching movies from Neo4j DB
    movies = get_movie_plots(limit=limit)

    print(len(movies))
    
    llm = OpenAI()
    # use OpenAI to generate an embedding (which is a jsonlist string) for each movie
    # then output it as a new row of data in the csv file
    for movie in movies:
        plot = f"{movie['title']}: {movie['tagline']}"

        response = llm.embeddings.create(
            input=plot,
            model='text-embedding-ada-002'
        )

        output_plot.writerow({
            'movieId': movie['movieId'],
            'embedding': response.data[0].embedding
        })

    csvfile_out.close()


def execute_cypher_query(limit=None, query=""):
    """
    connect to Neo4j DB and execute the provided query
    """
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )

    driver.verify_connectivity()

    query = query

    if limit is not None:
        query += f' LIMIT {limit}'

    result, summary, keys = driver.execute_query(
        query
    )

    print((result, summary, keys))

    driver.close()

    return True


# use OpenAI to generate a vector embedding for the tagline property (if it exists) of movie Nodes in Neo4j
# then save them as rows in a csv file, which could be loaded into Neo4j for further processing
generate_embeddings('./data/movie-plot-embeddings.csv')

# load OpenAI vector embedding as a new vector property onto the movie Node in Neo4j (by reading/loading the csv file)
execute_cypher_query(query="""
    LOAD CSV WITH HEADERS
    FROM 'file:///movie-plot-embeddings.csv'
    AS row
    MATCH (m:Movie)
    WHERE id(m) = toInteger(row.movieId)
    CALL db.create.setNodeVectorProperty(m, 'taglineEmbedding', apoc.convert.fromJsonList(row.embedding))
    RETURN count(*)
""")

# create a vector index in Neo4j for this new vector property that's used for the embedding, so as to search across these embeddings
execute_cypher_query(query="""
    CREATE VECTOR INDEX movieTaglineIdx IF NOT EXISTS
    FOR (m:Movie)
    ON m.taglineEmbedding
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }}
""")

# finally query the created vector index so to find the closest embedding matches to a given embedding
execute_cypher_query(query="""
    MATCH (m:Movie)
    WHERE m.taglineEmbedding IS NOT NULL AND m.title = "The Replacements"
    CALL db.index.vector.queryNodes('movieTaglineIdx', 3, m.taglineEmbedding)
    YIELD node, score
    RETURN node.title AS title, node.tagline AS tagline, score
""")
