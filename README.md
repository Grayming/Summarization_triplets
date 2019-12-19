# Unsupervised Multi-document summarization with graph clustering


## Methodology
Step 1: Fine tune a langauge model with "run_lm_finetuning.py".

Step 2: Text processing. See "IE.py".

Step 3: Graph construction. See "Graph.py".  Edge connection consider such factors: ADG(Refer to https://homes.cs.washington.edu/~mausam/papers/naacl13.pdf, https://www.aclweb.org/anthology/K17-1045.pdf), similar subject types. Weight is object similarities.

Step 4: Graph custering with graph_clustering in "Graph.py", use the graclus_cluster algorithm.  (Refer to http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf).

Step 5: Sentence compression (title generation) within each cluster. See "takahe.py". (Refer to https://www.aclweb.org/anthology/N13-1030.pdf, https://www.aclweb.org/anthology/C10-1037.pdf)


To run the whole framework, 'run_main.py'. To be done for whole flow.
## Datasets

Multi-News: https://github.com/Alex-Fabbri/Multi-News
