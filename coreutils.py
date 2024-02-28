#!/usr/bin/python
# -*- coding: utf-8 -*-

""" coreutils module: Provides common utilities for other modules """
from pathlib import Path
from time import sleep
import sys
from datetime import datetime, timezone
import json
from base64 import b64encode

import graphviz

import numpy as np
import psycopg

import spacy

import torch
import transformers
from sentence_transformers import SentenceTransformer

from coreconfigs import _SPACYMDL, _LLM_NAME, _LLM_MSG_TMPLT, _EMBED_MDL, _TXTSREADDIR, \
                        _DB_EMBED_DIM, _MAX_SIM_TXTS, _MAX_TKNLEN, _SPACY_MAX_TKNLEN, \
                        _PGHOST, _PGPORT, _PGUSER, _PGDB, _PGPWD


class DbOps():
    """ For database operations """
    def __init__(self):
        self.stmt = ''
        self.values = ''
        self._tmr = 0
        self._conn = ''
        self._dbconn_retry()

    def _dbconn_retry(self):
        try:
            self._conn = psycopg.connect(dbname=_PGDB,
                                        user=_PGUSER,
                                        password=_PGPWD,
                                        host=_PGHOST,
                                        port=_PGPORT,
                                        sslmode="prefer",
                                        connect_timeout=2)
        except psycopg.OperationalError:
            if self._tmr < 6:
                self._tmr += 3
                ### Server connection issue, try in few secs
                print(f"Unable to connect to database, trying in {self._tmr} secs...")
                sleep(self._tmr)
                self._dbconn_retry()
            else:
                self._tmr = 0
                raise

    def execstmt(self):
        """ Execute the DB statements """
        cur = self._conn.execute(self.stmt, self.values)
        res = ''
        if cur.description:  #check for return rows
            res = cur.fetchall()
        return res

    def commit(self):
        """ Commits the transaction"""
        self._conn.commit()

    def rollback(self):
        """ Rollback the transaction"""
        self._conn.rollback()


class Embeds():
    """
    Provides helper functions to
    1. Iterate all the directories under _TEXTDIR
       Read text, chunk and save in pgvector DB
    2. Generate embedding and store in pgvector DB
    3. If text_relations = True, then for each sentence generate <subj><relation>obj> triplet
    4. search for similar texts in pgvector DB
    """

    def __init__(self, dbconn=True, text_relations=True):
        self.emb_mdl = SentenceTransformer(_EMBED_MDL)
        self._text_relations = text_relations

        ## Verify embedding dimension size before processing
        embeddings = self.emb_mdl.encode("Hello World")
        if _DB_EMBED_DIM < embeddings.size:
            print(f"DB field length={_DB_EMBED_DIM}. Embedding dimension={embeddings.size}")
            print("Choose a different model or change embedding dimension on DB.")
            print("Exiting...")
            sys.exit(1)
        else:
            print("Embedding model ok.")
        if dbconn:
            self.dbo = DbOps()
            print("DB connection established.")
        # similarity: <=> cosine, <-> L2, <#> inner product
        # We normalize embeddings so use <#>
        # Ensure t_document_chunks index is using vector_ip_ops
        self.dbo_stmts = {"upd_doc":"update t_documents set created_at=%s where id=%s",
                     "ins_doc": "insert into t_documents (doc_name) values(%s) RETURNING id",
                     "sel_doc": "select id from t_documents where doc_name = %s",
                     "doc_refs": "select doc_name, doc_reference from t_documents \
                                    where id in ({qargs}) and doc_reference is NOT NULL",
                     "doc_images": "select img_desc, img_reference from t_document_images \
                                    where doc_id in ({qargs}) ",
                     "del_txts": "delete from t_document_chunks where doc_id = %s",
                     "del_relations": "delete from t_chunk_relations where doc_id = %s",
                     "ins_txt": "insert into t_document_chunks (doc_id, chunk, embedding) \
                                 values(%s, %s, %s) RETURNING id",
                     "ins_relations": "insert into t_chunk_relations (doc_id, chunk_id, text_relation, json_relation) \
                                 values(%s, %s, %s, %s)",
                     "ins_relations_nj": "insert into t_chunk_relations (doc_id, chunk_id, text_relation) \
                                 values(%s, %s, %s)",
                     "sim_chunks": "select json_relation from t_chunk_relations \
                                    where chunk_id in ({qargs}) and json_relation is NOT NULL",
                     "sim_txts": f"SELECT id, doc_id, chunk FROM t_document_chunks \
                                ORDER BY embedding <#> %s LIMIT {_MAX_SIM_TXTS}"
                    }
        if self._text_relations:
            self.prsr = spacy.load(_SPACYMDL)
            self.llm = LLMOps()
            self.llm.gconfigdct["temperature"] = .1
            self.llm.gconfigdct["max_new_tokens"] = 512
            system_content = """Translate the user content as entity relation triplet in 
                                {"subj": "", "relation": "", "obj": ""} json format."""
            self.llm.msg_tmplt[0]['content'] = system_content
            self.gconfig = transformers.GenerationConfig(**self.llm.gconfigdct)

    def np_to_str(self, val):
        """Convert np.float32 to np.float64. json.dumps supports it."""
        return np.float64(val)

    def dbexec(self, stmt, values, msg):
        """
        Generic function for executing database statements
        If results=False, returns ''
        If results=True returns all rows
        """
        self.dbo.stmt = stmt
        self.dbo.values = values
        retval = ''
        try:
            retval = self.dbo.execstmt()
        except Exception:
            print(f"{msg}  failed....")
            print("Rolling back transaction")
            print(f"Statement: {self.dbo.stmt}")
            print(f"Values: {self.dbo.values}")
            self.dbo.rollback()
            raise
        self.dbo.stmt = ''
        self.dbo.values = ''
        return retval

    def save_chunk_relations(self, txtlst, docid, chunkid):
        """ Get the relation triplet from LLM and insert into DB """
        def process_itm(jsn):
            json_val = ''
            if isinstance(jsn["obj"], str):
                if jsn.get("obj_qualifier"):
                    jsn["relation"] = f'{jsn["obj_qualifier"]} {jsn["relation"]}'
                if jsn.get("context"):
                    jsn["relation"] = f'{jsn["subj"]} {jsn["relation"]}'
                    jsn["subj"] = jsn["context"]
                if jsn["obj"]:
                    json_val = jsn
            elif isinstance(jsn["obj"], list):
                obj = ''
                try:
                    if isinstance(jsn["obj"][0], str):
                        obj = ', '.join(jsn["obj"])
                    elif isinstance(jsn["obj"][0], dict):
                        obj = ', '.join({i["subj"] for i in jsn["obj"] if i["subj"]})
                    if obj:
                        jsn["obj"] = obj
                        json_val = jsn
                except TypeError:
                    print("Ignoring list triplet due to incorrect json format")
            return json_val

        for text in txtlst:
            doc = self.prsr(text)
            pos = {tkn.pos_ for tkn in doc}
            # Generate relations only on sentences
            # with < 25 (default) tokens, else the generated relations can be too complicated.
            # with Noun and Verb
            if len(doc) < _SPACY_MAX_TKNLEN and 'NOUN' in pos and 'VERB' in pos:
                self.llm.msg_tmplt[1]['content'] = text
                print(f"Get relations: {text}")
                prompt = self.llm.pipeline.tokenizer.apply_chat_template(self.llm.msg_tmplt,
                                                                        tokenize=False,
                                                                        add_generation_prompt=False)
                outputs  = self.llm.pipeline(prompt, generation_config=self.gconfig)
                res = outputs[0]["generated_text"].split("<|assistant|>\n")[1]
                print(f"Generated triplet: {res}")
                jlst = []
                try:
                    jsn = json.loads(res)
                except json.decoder.JSONDecodeError:
                    try:
                        for itm in res.split('{')[1:]:
                            jsn = json.loads('{'+itm.replace('\n','').strip(','))
                            jitm = process_itm(jsn)
                            if jitm:
                                jlst.append(jitm)
                    except json.decoder.JSONDecodeError:
                        print("Ignoring triplet due to incorrect json format")
                else:
                    jitm = process_itm(jsn)
                    if jitm:
                        jlst.append(jitm)
                if jlst:
                    _ = self.dbexec(self.dbo_stmts['ins_relations'],
                    (docid, chunkid, res, json.dumps(jlst)),
                    "Insert chunk relations")
                else:
                    _ = self.dbexec(self.dbo_stmts['ins_relations_nj'],
                    (docid, chunkid, res),
                    "Insert chunk relations")    

    def save_embeddings_to_db(self, fldr, parent='.'):
        """
        Iterate all the directories under _TEXTDIR (fldr)
        Read text file, chunk texts and save chunk+embeddings in pgvector DB
        For each sentence get the relation triplet <subj> <relation> <obj>, 
        LLM optionally provides "obj_qualifier" or "context"
        """
        def emb_to_db(txtchunk, txtlst, docid):
            embeddings = self.emb_mdl.encode(txtchunk)
            # Normalizing the embeddings, just in case
            # default is Frobenius norm
            # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
            fnorm = np.linalg.norm(embeddings)
            lst = list(embeddings/fnorm)
            # json supports only np.float64. Convert np.float32
            embed_str = json.dumps(lst, default=np.float64)
            chunkid = self.dbexec(self.dbo_stmts['ins_txt'],
                            (docid, json.dumps(txtlst), embed_str),
                            "Insert chunk into Document")
            #print(f"docid: {docid}, chunkid:{chunkid[0][0]}")
            if self._text_relations:
                self.save_chunk_relations(txtlst, docid, chunkid[0][0])

        for rfl in fldr.iterdir():
            if rfl.is_file():
                print(f"Processing text file: {rfl.name}")
                # If the file has been processed already, delete the document chunks and reprocess
                doc_id = self.dbexec(self.dbo_stmts['sel_doc'], (rfl.name, ),
                                    "Check for Document")

                if doc_id:
                    docid = doc_id[0][0]
                    _ = self.dbexec(self.dbo_stmts['del_relations'], (docid, ),
                                    "Deleting chunk relations")
                    _ = self.dbexec(self.dbo_stmts['del_txts'], (docid, ),
                                    "Deleting document chunks")
                    _ = self.dbexec(self.dbo_stmts['upd_doc'],
                                    (datetime.now(tz=timezone.utc), docid),
                                    "Updating document timestamp")
                else:
                    doc_id = self.dbexec(self.dbo_stmts['ins_doc'], (rfl.name, ),
                                        "Insert Document")
                    docid = doc_id[0][0]
                with open(rfl, encoding="utf-8", errors="replace") as txt_fl:
                    filetexts = txt_fl.readlines()
                txtchunk = ''
                txtlst = []
                for txt in filetexts:
                    txt = txt.strip()
                    txtchunk = f"{txtchunk} {txt}"
                    txtlst.append(txt)
                    if len(txtchunk.split()) >= _MAX_TKNLEN:
                        emb_to_db(txtchunk, txtlst, docid)
                        txtlst = []
                        txtchunk = ''
                emb_to_db(txtchunk, txtlst, docid)  # Pending will be a separate chunk
                self.dbo.commit()
                print(f"Embeddings commited for file: {rfl}")
                try:
                    _ = rfl.replace(Path(_TXTSREADDIR, parent, rfl.name))
                except (PermissionError, FileExistsError, FileNotFoundError) as err:
                    print(f"File not moved: {err}")
                    print("Ignoring error...")

            if rfl.is_dir():
                print(f"Creating text processed directory: {rfl.name}")
                Path(_TXTSREADDIR, rfl.name).mkdir(parents=True, exist_ok=True)
                self.save_embeddings_to_db(rfl, rfl.name)
                # Delete the processed text directory, ignore error if any file exists
                try:
                    rfl.rmdir()
                except (OSError, FileNotFoundError) as err:
                    print(f"Directory not deleted: {err}")
                    print("Ignoring error...")

    def get_similar_texts(self, text):
        """
        1. Generate text embedding.
        2. Compare similarity against vectorDB and get texts similar to the input text.
        """
        embeddings = self.emb_mdl.encode(text)
        # Normalize before querying the DB
        fnorm = np.linalg.norm(embeddings)
        lst = list(embeddings/fnorm)
        # json supports only np.float64. Convert np.float32
        embed_str = json.dumps(lst, default=np.float64)
        sim_txts = self.dbexec(self.dbo_stmts['sim_txts'], (embed_str,), "Get similar texts")
        sim_chunk_ids = {itm[0] for itm in sim_txts}
        sim_doc_ids = {itm[1] for itm in sim_txts}
        all_txts = []
        contxt = ''
        # Avoid duplicate sentences, less noise in context is better for LLM response
        for itm in sim_txts:
            for txt in itm[2]:
                if txt not in all_txts:
                    all_txts.append(txt)
                    contxt = f"{contxt} {txt}"
                    # Do not exceed the tokens limit
                    if len(contxt.split()) >= _MAX_TKNLEN*_MAX_SIM_TXTS:
                        break
        return contxt, sim_chunk_ids, sim_doc_ids

class LLMOps():
    """For LLM operations """
    def __init__(self):
        self.pipeline = transformers.pipeline("text-generation",
                                              model=_LLM_NAME,
                                              torch_dtype=torch.bfloat16,
                                              device_map="auto",
                                             )
        self.gconfigdct = self.pipeline.model.generation_config.to_dict()
        self.gconfigdct["max_new_tokens"] =256
        self.gconfigdct["do_sample"] = True
        self.gconfigdct["top_k"] = 50
        self.gconfigdct["top_p"] = 0.95
        self.gconfigdct["pad_token_id"] = self.pipeline.model.config.eos_token_id
        self.gconfigdct["temperature"] = .7
        self.emb = ''
        self.msg_tmplt = _LLM_MSG_TMPLT

    def mdl_response(self, qry, temp=1):
        """ Function returns the answer from the LLM with context"""
        if temp < 1 or temp > 9:
            temp = 7
        if not self.emb:
            self.emb = Embeds(text_relations=False)
        contxt, sim_chunk_ids, sim_doc_ids = self.emb.get_similar_texts(qry)
        self.msg_tmplt[1]['content'] = contxt
        prompt = self.pipeline.tokenizer.apply_chat_template(self.msg_tmplt, tokenize=False,
                                                             add_generation_prompt=True)
        self.gconfigdct["temperature"] = temp/10
        gconfig = transformers.GenerationConfig(**self.gconfigdct)
        outputs = self.pipeline(prompt, generation_config=gconfig)
        res = outputs[0]["generated_text"].split("<|assistant|>\n")[1]
        #print(f"Similar Chunk ids: {sim_chunk_ids}")
        #print(f"Similar Doc ids: {sim_doc_ids}")

        # Get document relations
        dbqry = self.emb.dbo_stmts['sim_chunks']
        dbqry = dbqry.replace("{qargs}", ','.join(str(i) for i in sim_chunk_ids))
        dbres = self.emb.dbexec(dbqry, None, "Get chunk relations")
        sim_chunk_lst = []
        for each in dbres:
            for row in each:
                for itm in row:
                    sim_chunk_lst.append((itm["subj"], itm['obj'], itm['relation']))

        # Get document references, document images
        docs = ','.join(str(i) for i in sim_doc_ids)
        dbqry = self.emb.dbo_stmts['doc_refs']
        dbqry = dbqry.replace("{qargs}", docs)
        dbres = self.emb.dbexec(dbqry, None, "Get document references")
        sim_doc_refs = {row[0]:row[1] for row in dbres}

        dbqry = self.emb.dbo_stmts['doc_images']
        dbqry = dbqry.replace("{qargs}", docs)
        dbres = self.emb.dbexec(dbqry, None, "Get document images")
        sim_doc_imgs = {row[0]:row[1] for row in dbres}
        return res, set(sim_chunk_lst), sim_doc_refs, sim_doc_imgs

    def mdl_ui_response(self, qry, temp=1):
        """ Function returns ui friendly results from the LLM 
        Returns a dictionary of UI elements {"answer", "graph", "images", "docs"}
        Query Answer, relations graph, images from documents, associated documents
        """
        res, sim_chunk_lst, sim_doc_refs, sim_doc_imgs = self.mdl_response(qry, temp)
        # Build graph, return as jpeg image
        grph = graphviz.Digraph('wide')
        for row in sim_chunk_lst:
            grph.edge(row[0].lower(), row[1].lower(), row[2].lower())
        unflt = grph.unflatten(stagger=5)
        grph_html = "<h2>Relations graph</h2><div style='max-width:100%; max-height:720px; overflow:auto'>"
        grph = '<img src="data:image/jpeg;base64,%s"</img></div>'
        grph_html +=  grph %(b64encode(unflt._repr_image_jpeg()).__repr__()[2:-1])

        # Images in document
        img_html = '<h2>Images in documents</h2>'
        img = '<div><p>%s<img src="%s" alt="%s"></div>'
        for key, val in sim_doc_imgs.items():
            img_html += img %(key, val, key)

        # Documents queried for context
        doc_html = '<h2>Documents referenced </h2>'
        ref = '<div><p><a href="%s" target="_blank" title="%s">%s</a></div>'
        for key, val in sim_doc_refs.items():
            doc_html +=  ref %(val, val, key.split('.')[0])
        return {"answer":res, "graph":grph_html, "images":img_html, "docs":doc_html}
