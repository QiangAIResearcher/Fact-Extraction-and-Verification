# Fact-Extraction-and-Verification

This is the repository for the task of Fact Extraction and Verification described in the NAACL2018 paper: : [FEVER: A large-scale dataset for Fact Extraction and VERification.]()

>Unlike other tasks and despite recent interest, research in textual claim verification has been hindered by the lack of large-scale manually annotated datasets. In this paper we introduce a new publicly available dataset for verification against textual sources, FEVER: Fact Extraction and VERification. It consists of 185,441 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The claims are classified as Supported, Refuted or NotEnoughInfo by annotators achieving 0.6841 in Fleiss κ. For the first two classes, the annotators also recorded the sentence(s) forming the necessary evidence for their judgment. 

## Data Preparation

Download the FEVER dataset from the [website](http://fever.ai/data.html) into the data directory

    mkdir data
    mkdir data/fever-data
    
    # We use the data used in the baseline paper
    wget -O data/fever-data/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
    wget -O data/fever-data/dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl
    wget -O data/fever-data/test.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl
    
The data preparation consists of three steps: downloading the articles from Wikipedia, indexing these for the Evidence Retrieval and performing the negative sampling for training.  

### 1. Download Wikipedia data
Download the pre-processed Wikipedia articles from [the website](https://sheffieldnlp.github.io/fever/data.html) and unzip it into the data folder.
    
    wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
    unzip wiki-pages.zip -d data
 
### 2. Construct SQLite Database
Construct an SQLite Database. A commercial personal laptop seems not work when dealing with the entire database as a single file so we split the Wikipedia database into a few files too. 
    
    python build_db.py

### 3. Create Term-Document count matrices and merge
Create a term-document count matrix for each split, and then merge the count matrices.
    
    python build_count_matrix.py data/fever data/index
    python merge_count_matrix.py data/index data/index

## Baseline

 The baseline model constists of two Logistic Regression models based on TF-IDF for document retrieval and sentence selection respectively. 
 
    python reweight_count_matrix.py data/index/count-ngram\=1-hash\=16777216.npz data/index --model tfidf

The remaining task for FEVER challenge, i.e. document retrieval, sentence selection, sampling for NotEnoughInfo, and RTE training are done in IPython notebook `fever.ipynb` and implementation in `fever.py`. The class `Oracle` reads either TF-IDF or PMI matrix and have methods for finding relevant documents, sentences, etc. given the input claim.

## Find Out More

 Visit [http://fever.ai](http://fever.ai) to find out more about the shared task and download the data.
