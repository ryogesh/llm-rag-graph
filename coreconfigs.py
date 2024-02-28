#!/usr/bin/python
# -*- coding: utf-8 -*-

""" All configuration items
 Important: CHANGE the postgresDB connection information below
 Optional: To ignore during text extraction, Change
            _IGNORE_SENTS: list of sentences (not words)
"""

# Models used: Embedding model, LLM and Spacy model (for sentence identification)
# Embedding Model with lesser Dimension=384, better for vectorDB performance
# Use MTEB leaderboard: https://huggingface.co/spaces/mteb/leaderboard for model details
# Embedding Model Sequence Length = 512, setting _MAX_TKNLEN to about 25%
# Text chunks shouldn't be too short or too long to be of good context
_EMBED_MDL = "khoa-klaytn/bge-small-en-v1.5-angle"
_DB_EMBED_DIM = 384
_MAX_TKNLEN = 120

# LLM
_LLM_NAME = "HuggingFaceH4/zephyr-7b-beta"
_LLM_MSG_TMPLT = [{ "role": "system", "content": "",}, {"role": "user", "content": ''},]

# LLM model sequence length = 4k, we will provide about 1k tokens, _MAX_TKNLEN*_MAX_SIM_TXTS
# Higher length requires higher GPU processing, memory and can lead to OoM error on smaller GPUs.
# Reducing context tokens, reduces processing costs.
# But short contexts may lead to inaccurate or repetitive answers.
_MAX_SIM_TXTS = 4

# SciSpacy model
# see https://github.com/allenai/scispacy?tab=readme-ov-file#available-models
_SPACYMDL = "en_core_sci_lg"
_SPACY_MAX_TKNLEN = 25

# Directory to store extracted texts
_TEXTDIR = "texts_input"
# Directory to store texts once embeddings are stored in vector DB
_TXTSREADDIR = "texts_processed"


#PgVector DB details
_PGHOST = "1.1.1.1"
_PGPORT = 5432
_PGUSER = "ragu"
_PGDB = "ragdb"
_PGPWD = "yourpassword"
