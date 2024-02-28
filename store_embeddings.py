#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Script to 
1. Iterate all files under the directory
2. Read text file, chunk texts and save chunk+embeddings in pgvector DB
"""

from pathlib import Path

from coreconfigs import _TEXTDIR
from coreutils import Embeds


texts = Path(_TEXTDIR)
embd = Embeds(dbconn=True, text_relations=True)
embd.save_embeddings_to_db(texts)
