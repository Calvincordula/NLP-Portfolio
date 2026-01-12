# NLP Portfolio

This repository contains Jupyter notebooks on the basics of Natural Language Processing through hands-on projects.
- Linguistic Analysis: Named Entity Recognition (NER), Part-of-Speech (POS) Tagging, Text Tagging, Entity Extraction
- Semantics & Modeling: Sentiment Analysis, Topic Modeling, Text Vectorization (TF-IDF, Word Embeddings), Feature Engineering
- Preprocessing: Tokenization, Normalization, Stop word removal, Lemmatization, Stemming, Data Augmentation


# Installed Packages

These are the installed packages and versions used for this course. All were installed in a conda environment (see below for how I created this).

python=3.11

nltk==3.9.1 
pandas==2.2.3 
matplotlib==3.10.0 
spacy==3.8.3 
textblob==0.18.0.post0 
vaderSentiment==3.3.2 
transformers==4.47.1 
scikit-learn==1.6.0 
gensim==4.3.3 
seaborn==0.13.2 
torch==2.5.1 
ipywidgets==8.1.5


# Conda Environment

conda create --name nlp_env python=3.11
conda activate nlp_env
pip install nltk==3.9.1 pandas==2.2.3 matplotlib==3.10.0 spacy==3.8.3 textblob==0.18.0.post0 vaderSentiment==3.3.2 transformers==4.47.1 scikit-learn==1.6.0 gensim==4.3.3 seaborn==0.13.2 torch==2.5.1 ipywidgets==8.1.5
python -m spacy download en_core_web_sm
pip install ipykernel jupyterlab notebook
python -m ipykernel install --user --name=nlp_env
