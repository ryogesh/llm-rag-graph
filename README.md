# Build relationship Graphs using LLM in a Retrieval-Augmented Generation(RAG) framework with pgvector as a vector database


## Overview

Tool to build relationship graphs using a large language module (LLM). 
Supports adding context to the query using Retrieval-Augmented Generation(RAG). Context is built against an internal knowledge base. Context embeddings are stored and retrieved from a vector database. Relationships are stored in the database.


## Tool Features
- Store context in the vector database
- Retrieve context from vector database, supplement the query with the context thus improve LLM response quality
- Along with the LLM response, visualize the relationships in the document(s), highlight related documents and images


## Installation
### Prerequisites

- [Python](https://www.python.org/downloads/) 3.10 or greater
- check requirements.txt for required python libraries

### Supported Database

- [PostgreSQL](https://www.postgresql.org/) . Supports Postgres 11+ . Tested on 14.10.

### Vector Database

- [pgvector](https://github.com/pgvector/pgvector) 


### Scripts

- pgdb_setup.sh: Install postgresql14.10 database on Ubuntu.
- pgvector.sql: Configure postgresql database as a vector database
- setup.sh: Install required python packages, configure vector database. Assumes PostgreSQL database on the same host. Review the file before execution.


## Application

- coreconfigs.py: Application configurations. An important file to review and edit.
- store_embeddings.py: Wrapper script to read the text files, generate and store embeddings, relationships in pgvector database
- example_query.py: Example to query LLM, save results as a html
- LLM-RAG-GRAPH.ipynb: Jupyter notebook with Gradio interface can also be used to interact with the LLM and visualize the graph


## Getting Started

### Application config and run
- Download the repo
- Perform the installation steps (see above)
- #### Edit coreconfigs.py to update the postgreSQL DB connection.

- run store_embeddings.py to store the embeddings, relationships into pgvector DB

    ```
	Embedding model ok.
	DB connection established.
	Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 8/8 [00:02<00:00,  3.39it/s]
	Processing text file: NBK548420.txt
	Get relations: Cetirizine and its enantiomer levocetirizine are second generation antihistamines that are used for the treatment of allergic rhinitis, angioedema and chronic urticaria.
	...
	...
	Embeddings commited for file: texts_input\NBK548420.txt
    ```

- run the example_query.py to test

    ```
    python example_query.py
	Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 8/8 [00:04<00:00,  1.99it/s]
	WARNING:root:Some parameters are on the meta device device because they were offloaded to the cpu.
	Embedding model ok.
	DB connection established.
	View the html file: user_qry_results.html for the results
    
	...
    ```


## Example 1

<div align="center">
	<img src=assets/Q1.jpg width=90% />
	<h4> Generated graph full resolution</h4>
	<img src=assets/Q1-graph.jpg width=90% />
</div>


## Example2: Query with a typo

<div align="center">
	<img src=assets/Q2.jpg width=90% />
	<h4> Generated graph full resolution</h4>
	<img src=assets/Q2-graph.jpg width=90% />
</div>