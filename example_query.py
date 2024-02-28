#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Example script to query LLM """

from coreutils import LLMOps


llm = LLMOps()

#typo in the query
user_qry = "cetrizine guideline"
dct = llm.mdl_ui_response(user_qry)

html = f'<html><body><div><p><h3>Question:</h3>{user_qry}\
         <p><h3>LLM Response:</h3>{dct["answer"]}</div>\
         {dct["images"]} {dct["docs"]} {dct["graph"]} </body></html>'
fname = "user_qry_results.html"
with open(fname, 'wt', encoding="utf-8") as fl:
    fl.write(html)

print(f"View the html file: {fname} for the results")
