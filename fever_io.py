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

def save_jsonl(dictionaries, path, print_message=True):
    """save jsonl file from list of dictionaries
    """
    if os.path.exists(path):
        raise OSError("file {} already exists".format(path))

    if print_message:
        print("saving at {}".format(path))
    with open(path, "a") as out_file:
        for instance in dictionaries:
            out_file.write(json.dumps(instance) + "\n")


def read_jsonl(path):
    with open(path, "r") as in_file:
        out = [json.loads(line) for line in in_file]
    return out


def parse_wiki(wikipedia_dir, doc_id_dir):
    """
    Returns a dictionary lookup from document id (URL) to document content.
    Saves the lookup in ../data/doc_id_text to speed up subsequent passes.
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
    """load all lines for provided titles
    Args
    titles: list of titles
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
    """Returns a dictionary from titles to line numbers to line text.
    Args
    docs: {claim_id: [(title, sentence_num),  ...], ...}

    Input is a dictionary from claim ids to titles and line numbers,
    and a lookup from titles to filenumbers.
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
    """lookup corresponding sentences and return list of sentences
    Args
    evidences: [(title, linum), ...]
    t2l2s: title2line2sentence <- output of load_doc_lines

    Returns
    list of evidence sentences
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

# def term_and_doc_freqs():
#     ## calculate term frequency and document frequency
#     df=Counter()
#     df0=Counter()
#     with open("../data/tf","w") as f:
#         with open("../data/tf_pointers","w") as p:
#             for filename in tqdm(sorted(os.listdir("../data/wiki-pages/wiki-pages/"))):
#                 with open("../data/wiki-pages/wiki-pages/"+filename, 'r') as openfile:
#                     for line in openfile:
#                         data=json.loads(line.rstrip("\n"))
#                         title=data["id"]
#                         tf=Counter()
#                         tf0=Counter()
#                         tset=set()
#                         t0set=set()
#                         lines=data["lines"]
#                         for l in lines.split("\n"):
#                             fields=l.split("\t")
#                             if fields[0].isnumeric():
#                                 l_id=int(fields[0])
#                                 l_txt=fields[1]
#                                 toks=word_tokenize(l_txt.lower())
#                                 for tok in toks:
#                                     if l_id==0:
#                                         tf0[tok]+=1
#                                         t0set.add(tok)
#                                     else:
#                                         tf[tok]+=1
#                                         tset.add(tok)
#                         for tok in tset:
#                             df[tok]+=1
#                         for tok in t0set:
#                             df0[tok]+=1
#                         if title != "":
#                             point=f.tell()
#                             f.write(title+"\n")
#                             p.write(title+"\t"+str(point))
#                             terms=0
#                             for tok,c in tf.most_common():
#                                 if tok != "":
#                                     c0=tf0[tok]
#                                     del tf0[tok]
#                                     f.write(" "+tok+" "+str(c)+" "+str(c0)+"\n")
#                                     terms+=1
#                             for tok,c0 in tf0.most_common():
#                                 if tok != "":
#                                     c=tf[tok]
#                                     f.write(" "+tok+" "+str(c)+" "+str(c0)+"\n")
#                                     terms+=1
#                             p.write("\t"+str(terms)+"\n")
#     with open("../data/df","w") as f:
#         for tok,c in df.most_common():
#             if tok != "":
#                 c0=df0[tok]
#                 del df0[tok]
#                 f.write(tok+" "+str(c)+" "+str(c0)+"\n")
#         for tok,c0 in df0.most_common():
#             if tok != "":
#                 c=df[tok]
#                 f.write(tok+" "+str(c)+" "+str(c0)+"\n")

# def titles_to_tf(tf_pointers="../data/tf_pointers"):
#     ## convert titles to term frequency representation
#     t2tf=dict()
#     with open(tf_pointers) as f:
#         for line in f:
#             fields=line.rstrip("\n").split("\t")
#             title=fields[0]
#             point=int(fields[1])
#             terms=int(fields[2])
#             t2tf[title]=(point,terms)
#     return t2tf

# def load_doc_tf(docs=dict(),t2tf=dict(),term_freqs="../data/tf"):
#     ## convert document to term frequenct representation
#     doctf=dict()
#     toks=dict()
#     points=set()
#     for cid in docs:
#         titles, ctoks = docs[cid]
#         for title in titles:
#             doctf[title]=dict()
#             if title not in toks:
#                 toks[title]=set()
#             for tok in ctoks:
#                 toks[title].add(tok)
#             point,terms=t2tf[title]
#             points.add(point)
#     points=sorted(list(points))
#     with open(term_freqs) as f:
#         for point in points:
#             f.seek(point,0)
#             line=f.readline()
#             title=line.rstrip("\n")
#             p,terms=t2tf[title]
#             for i in range(terms):
#                 line=f.readline()
#                 fields=line.lstrip(" ").split()
#                 tok=fields[0]
#                 if tok in toks[title]:
#                     tf=int(fields[1])
#                     tf0=int(fields[2])
#                     doctf[title][tok]=(tf,tf0)
#     return doctf

# def load_split_trainset(dev_size:int):
#     """
#     Loads the full training set, splits it into preliminary train and dev set.
#     This preliminary dev set is balanced.
#     dev_size: size of dev set.
#     """
#
#     # load fever training data
#     full_train = load_dataset_json()
#
#     positives = []
#     negatives = []
#     neutrals = []
#
#     # sort dataset according to label.
#     for example in full_train:
#         example['id']
#         label = example['label']
#         if label == "SUPPORTS":
#             positives.append(example)
#         elif label == "REFUTES":
#             negatives.append(example)
#         elif label == "NOT ENOUGH INFO":
#             neutrals.append(example)
#         else:
#             raise AssertionError("Bad label!", label)
#
#     # shuffle examples for each label.
#     random.seed(42)
#     random.shuffle(positives)
#     random.shuffle(negatives)
#     random.shuffle(neutrals)
#
#     # split off a preliminary dev set, balanced across each of the three classes
#     size = int(dev_size/3)
#     preliminary_dev_set = positives[:size] + negatives[:size] + neutrals[:size]
#
#     # the remaining data will be the new training data
#     train_set = positives[size:] + negatives[size:] + neutrals[size:]
#
#     # shuffle order of examples
#     random.shuffle(preliminary_dev_set)
#     random.shuffle(train_set)
#
#     return train_set, preliminary_dev_set
# def get_label_set():
#     label_set = {"SUPPORTS","REFUTES","NOT ENOUGH INFO"}
#     return label_set

# def load_wikipedia(wikipedia_dir="../data/wiki-pages/wiki-pages/", instance_num=1e6):
#     """
#     Returns a list with in total 5,416,537 wikipedia article texts as elements.
#     If one doesn't want to load all articles, one can use "instance_num" to specify instance_num files should be
#     read (each containing 50000 articles). For example, to read only 100K articles, pick instance_num=2.
#     """
#     all_texts = []
#     print("loading wikipedia...")
#     for filename in tqdm(sorted(os.listdir(wikipedia_dir))[:instance_num]):
#         with open(wikipedia_dir+filename, 'r') as openfile:
#             some_texts = [json.loads(line)['text'] for line in openfile.readlines()]
#         all_texts.extend(some_texts)
#     print("Loaded", len(all_texts), "articles. Size (MB):", round(sys.getsizeof(all_texts)/1024/1024, 3))
#     return all_texts


def load_dataset_json(path, instance_num=1e6):
    """
    Reads the Fever Training set, returns list of examples.
    instance_num: how many examples to load. Useful for debugging.
    """
    data = []
    with open(path, 'r') as openfile:
        for iline, line in enumerate(openfile.readlines()):
            data.append(json.loads(line))
            if iline+1 >= instance_num:
                break
    return data

def load_dataset(set_type,instance_num=1e6):
    """Reads the Fever train/dev set used on the paper.
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
