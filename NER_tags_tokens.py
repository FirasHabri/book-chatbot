
import tensorflow as tf
import numpy as np
sess=tf.Session()    
saver = tf.train.import_meta_graph('my_test_model/mytestmodel.meta')
saver.restore(sess,tf.train.latest_checkpoint('my_test_model/'))
graph = tf.get_default_graph()
input_batch = graph.get_tensor_by_name("input_batch:0")
lengths = graph.get_tensor_by_name("lengths:0")
all_vars = tf.get_collection('vars')

def read_data(file_path):
    tokens = []
    tags = []
    
    book_tokens = []
    book_tags = []
    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            if book_tokens:
                tokens.append(book_tokens)
                tags.append(book_tags)
            book_tokens = []
            book_tags = []
        else:
            token, tag = line.split()
            book_tokens.append(token)
            book_tags.append(tag)
    return tokens, tags

train_tokens, train_tags = read_data('dataNER/train.txt')
validation_tokens, validation_tags = read_data('dataNER/validation.txt')
test_tokens, test_tags = read_data('dataNER/test.txt')

from collections import defaultdict

def build_dict(tokens_or_tags, special_tokens):
    tok2idx = defaultdict(lambda: 0)
    idx2tok = []
    cur_idx=0
    
    for s_token in special_tokens:
        if not s_token in tok2idx:
            tok2idx[s_token]=cur_idx
            cur_idx+=1
            idx2tok.append(s_token)

    for tokens in tokens_or_tags:
        for token in tokens:
            if not token in tok2idx:
                tok2idx[token]=cur_idx
                cur_idx+=1
                idx2tok.append(token)
    return tok2idx, idx2tok

special_tokens = ['<UNK>', '<PAD>']
special_tags = ['O']

token2idx, idx2token = build_dict(train_tokens + validation_tokens, special_tokens)
tag2idx, idx2tag = build_dict(train_tags, special_tags)

def words2idxs(tokens_list):
    return [token2idx[word] for word in tokens_list]

def tags2idxs(tags_list):
    return [tag2idx[tag] for tag in tags_list]

def idxs2words(idxs):
    return [idx2token[idx] for idx in idxs]

def idxs2tags(idxs):
    return [idx2tag[idx] for idx in idxs]

def predict_tags(token_idxs_batch,Outputlist):    
    tag_idxs_batch = Outputlist

    tags_batch, tokens_batch = [], []
    for tag_idxs, token_idxs in zip(tag_idxs_batch, token_idxs_batch):
        tags, tokens = [], []
        for tag_idx, token_idx in zip(tag_idxs, token_idxs):
            tags.append(idx2tag[tag_idx])
            tokens.append(idx2token[token_idx])
        tags_batch.append(tags)
        tokens_batch.append(tokens)
    return tags_batch, tokens_batch

def defStr(Str):
    string = Str
    return string