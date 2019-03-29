# Adapted and modified from https://github.com/sheffieldnlp/fever-baselines/tree/master/src/scripts
# which is adapted from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
import json
import random
import re
import os
import sys
from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize


def parse_wiki(wikipedia_dir, doc_id_dir):
    """
    This function traverses all the jsonl files
    and returns a dictionary containing document ID and corresponding content

    Args
    wikipedia_dir: the parent directory of the jsonl files
    doc_id_dir: the location of wiki-pages

    Returns
    a dictionary: document ID as dictionary keys and document content as values.
    
    Remark: Saves the dictionary in ../data/doc_id_text to speed up subsequent passes.
    """
    # doc_id_text saves the title and content of each wiki-page
    doc_id_text=dict()
    try:
        with open(doc_id_dir, "r") as f:
            print("Reading from" + str(doc_id_dir) )
            for line in f:
                fields=line.rstrip("\n").split("\t")
                doc_id=fields[0]
                text=fields[1]
                doc_id_text[doc_id]=text
    except:
        print(doc_id_dir)
        with open(doc_id_dir,"w") as w:
            print("Constructing " + str(doc_id_dir))
            for i in tqdm(range(1,110)):# jsonl file number from 001 to 109
                jnum="{:03d}".format(i)
                fname=wikipedia_dir+"wiki-"+jnum+".jsonl"
                with open(fname) as f:
                    # point=f.tell()# file pointer starting from 0
                    line=f.readline()
                    while line:
                        data=json.loads(line.rstrip("\n"))
                        doc_id=data["id"]
                        text = data["text"]
                        lines=data["lines"]
                        if text != "":
                            w.write(doc_id+"\t"+text+"\n")
                            doc_id_text[doc_id]=text
                        # point=f.tell()
                        line=f.readline()
        
    return doc_id_text


def load_doclines(titles, t2jnum, filtering=True):
    """
    This function loads all lines for provided document ID

    Args
    titles: list of document ID

    Remark: a document ID is the same as its title
    """
    if filtering:
        # select title from titles if this title is in the wiki-pages
        filtered_titles = [title for title in titles if title in t2jnum]
        print("mismatch: {} / {}".format(len(titles) - len(filtered_titles), len(titles)))
        titles = filtered_titles

    docs ={"dummy_id": [(title, "dummy_linum") for title in titles]}
    doclines = load_doc_lines(docs, t2jnum, wikipedia_dir="../data/wiki-pages/wiki-pages/")
    return doclines


def load_doc_lines(docs=dict(), t2jnum=dict(), wikipedia_dir="../data/wiki-pages/wiki-pages/"):
    """
    This function returns a dictionary from titles to line numbers to line text.

    Args
    docs: a dictionary from claim ids to titles and line numbers
    e.g., {claim_id: [(title, sentence_num),  ...], ...}


    t2jnum: a dictionary from titles to filenumbers.
    """
    doclines = dict()
    jnums = dict()
    titles = set()
    ## cid is the claim id that is an integer
    for cid in docs:
        for title, sentence_num in docs[cid]:
            doclines[title] = dict()
            titles.add(title)
            if title in t2jnum:
                jnum, point = t2jnum[title]
                if jnum not in jnums:
                    jnums[jnum] = set()
                jnums[jnum].add(point)
            else:
                print(str(title) + " not in t2jnum!")
    for jnum in tqdm(jnums):
        points = sorted(list(jnums[jnum]))
        fname = wikipedia_dir + "wiki-" + jnum + ".jsonl"
        with open(fname) as f:
            for point in points:
                f.seek(point, 0)
                line = f.readline()
                data = json.loads(line.rstrip("\n"))
                title = data["id"]
                lines = data["lines"]
                assert title in titles
                if title in titles and lines != "":
                    for l in lines.split("\n"):
                        fields = l.split("\t")
                        if fields[0].isnumeric():
                            l_id = int(fields[0])
                            l_txt = fields[1]
                            doclines[title][l_id] = l_txt
    return doclines


def get_evidence_sentence_list(evidences, t2l2s, prependlinum=False, prependtitle=False):
    """
    This function looks up corresponding evidence sentences and return list of sentences
    Args
    evidences: [(title, linum), ...]
    t2l2s: title2line2sentence <- output of load_doc_lines

    Return
    a list of evidence sentences
    """
    SEP = "#"
    def process_title(title):
        """ 'hoge_fuga_hoo' -> 'hoge fuga hoo' """
        return re.sub("_", " ", title)

    def maybe_prepend(title, linum):
        prep = list()
        if prependtitle:
            prep.append(title)
        if prependlinum:
            prep.append(str(linum))

        content = " {} ".format(SEP).join(prep)
        if prep:
            return "{0} {1} {0}".format(SEP, content)
        else:
            return content

    titles = [title for title, _ in evidences]
    linums = [linum for _, linum in evidences]

    return [ (maybe_prepend(process_title(title), linum) + " " + t2l2s[title][linum]).strip() for title, linum in zip(titles, linums)]



def load_dataset_json(path, instance_num=1e6):
    """
    Args
    path: the location of the data set tp read
    instance_num: how many instances to load. Useful for debugging

    Returns
    a list of instances
    """
    data = []
    with open(path, 'r') as openfile:
        for iline, line in enumerate(openfile.readlines()):
            data.append(json.loads(line))
            if iline+1 >= instance_num:
                break
    return data

def load_dataset(set_type,instance_num=1e6):
    """
    Args
    set_type: the train/dev set
    instance_num: the number of instances to read

    Return
    a list of instances by calling the function load_dataset_json
    """
    if set_type == 'train':
        dataset = load_dataset_json(path="/home/ubuntu/efs/fever_data/train.jsonl", instance_num=instance_num)
    if set_type == 'dev':
        dataset = load_dataset_json(path="/home/ubuntu/efs/fever_data/dev.jsonl", instance_num=instance_num)
    return dataset


if __name__ == "__main__":
    # load fever training data

    train_path = '../data/train.jsonl'
    train_data = load_dataset_json(path=train_path,instance_num=20)
    print(len(train_data))

    dev_path = '../data/dev.jsonl'
    dev_data = load_dataset_json(path=dev_path, instance_num=20)
    print(len(dev_data))


    for sample in train_data[:3]:
        print(sample)
