MATCH (m:Movie) 
WHERE toLower(m.title) CONTAINS "matrix"
RETURN m


LOAD CSV WITH HEADERS
FROM 'file:///movie-plot-embeddings.csv'
AS row
MATCH (m:Movie) 
WHERE id(m) = toInteger(row.movieId)
RETURN m


MATCH (n) 
WHERE n.taglineEmbedding IS NOT NULL
RETURN n


CREATE VECTOR INDEX movieTaglineIdx IF NOT EXISTS
FOR (m:Movie)
ON m.taglineEmbedding
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}


MATCH (m:Movie)
WHERE toLower(m.title) CONTAINS "matrix"
CALL db.index.vector.queryNodes('movieTaglineIdx', 6, m.taglineEmbedding)
YIELD node, score
RETURN node.title AS title, node.tagline AS tagline, score