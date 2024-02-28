
# Install required python packages
pip install -r requirements.txt

# Install scispacy model, refer https://github.com/allenai/scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz

# If postgresDB is not installed
pgdb_setup.sh

# Setup the vector database for RAG
su -c 'psql < pgvector.sql' postgres 
