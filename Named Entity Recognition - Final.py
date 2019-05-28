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

train_tokens, train_tags = read_data('data/train.txt')
validation_tokens, validation_tags = read_data('data/validation.txt')
test_tokens, test_tags = read_data('data/test.txt')

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

def batches_generator(batch_size, tokens, tags,
                      shuffle=True, allow_smaller_last_batch=True):
    
    n_samples = len(tokens)
    if shuffle:
        order = np.random.permutation(n_samples)
    else:
        order = np.arange(n_samples)

    n_batches = n_samples // batch_size
    if allow_smaller_last_batch and n_samples % batch_size:
        n_batches += 1

    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k + 1) * batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        x_list = []
        y_list = []
        max_len_token = 0
        for idx in order[batch_start: batch_end]:
            x_list.append(words2idxs(tokens[idx]))
            y_list.append(tags2idxs(tags[idx]))
            max_len_token = max(max_len_token, len(tags[idx]))
            
        x = np.ones([current_batch_size, max_len_token], dtype=np.int32) * token2idx['<PAD>']
        y = np.ones([current_batch_size, max_len_token], dtype=np.int32) * tag2idx['O']
        lengths = np.zeros(current_batch_size, dtype=np.int32)
        for n in range(current_batch_size):
            utt_len = len(x_list[n])
            x[n, :utt_len] = x_list[n]
            lengths[n] = utt_len
            y[n, :utt_len] = y_list[n]
        yield x, y, lengths
import tensorflow as tf
import numpy as np

class BiLSTMModel():
    pass

def declare_placeholders(self):

    self.input_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_batch') 
    self.ground_truth_tags =tf.placeholder(dtype=tf.int32, shape=[None, None], name='ground_truth_tags')
  
    self.lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths') 

    self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
    
    self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[])

BiLSTMModel.__declare_placeholders = classmethod(declare_placeholders)

def build_layers(self, vocabulary_size, embedding_dim, n_hidden_rnn, n_tags):
    
    initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
    embedding_matrix_variable = tf.Variable(initial_embedding_matrix,dtype=tf.float32)

    forward_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_rnn)
    forward_cell = tf.contrib.rnn.DropoutWrapper(forward_cell, output_keep_prob=self.dropout_ph)
    
    backward_cell =tf.contrib.rnn.BasicLSTMCell(n_hidden_rnn)
    backward_cell =tf.contrib.rnn.DropoutWrapper(backward_cell, output_keep_prob=self.dropout_ph)                                                     

    embeddings = tf.nn.embedding_lookup(embedding_matrix_variable,self.input_batch)

    (rnn_output_fw, rnn_output_bw), _ =tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                backward_cell,embeddings,sequence_length=self.lengths,dtype=tf.float32)
    rnn_output = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)

    self.logits = tf.layers.dense(rnn_output, n_tags, activation=None)

BiLSTMModel.__build_layers = classmethod(build_layers)

def compute_predictions(self):
    
    softmax_output = tf.exp(self.logits) / tf.reduce_sum(tf.exp(self.logits))
    
    self.predictions = tf.argmax(softmax_output,axis=-1)
BiLSTMModel.__compute_predictions = classmethod(compute_predictions)

def compute_loss(self, n_tags, PAD_index):

    ground_truth_tags_one_hot = tf.one_hot(self.ground_truth_tags, n_tags)
    loss_tensor =  tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=ground_truth_tags_one_hot)

    
    mask = tf.cast(tf.not_equal(self.input_batch, PAD_index), tf.float32)
    self.loss = tf.reduce_mean(loss_tensor*mask)
BiLSTMModel.__compute_loss = classmethod(compute_loss)

def perform_optimization(self):
    
    self.optimizer =  tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
    self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
    
    clip_norm = tf.cast(1.0, tf.float32)
    self.grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in self.grads_and_vars]
    
    self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)

BiLSTMModel.__perform_optimization = classmethod(perform_optimization)

def init_model(self, vocabulary_size, n_tags, embedding_dim, n_hidden_rnn, PAD_index):
    self.__declare_placeholders()
    self.__build_layers(vocabulary_size, embedding_dim, n_hidden_rnn, n_tags)
    self.__compute_predictions()
    self.__compute_loss(n_tags, PAD_index)
    self.__perform_optimization()

BiLSTMModel.__init__ = classmethod(init_model)
def train_on_batch(self, session, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability):
    feed_dict = {self.input_batch: x_batch,
                 self.ground_truth_tags: y_batch,
                 self.learning_rate_ph: learning_rate,
                 self.dropout_ph: dropout_keep_probability,
                 self.lengths: lengths}
    
    session.run(self.train_op, feed_dict=feed_dict)
BiLSTMModel.train_on_batch = classmethod(train_on_batch)

def predict_for_batch(self, session, x_batch, lengths):
    predictions=session.run(self.predictions,feed_dict={self.input_batch:x_batch,self.lengths:lengths})

    return predictions

BiLSTMModel.predict_for_batch = classmethod(predict_for_batch)

from evaluation import precision_recall_f1

def predict_tags(model, session, token_idxs_batch, lengths):
    
    tag_idxs_batch = model.predict_for_batch(session, token_idxs_batch, lengths)
    
    tags_batch, tokens_batch = [], []
    for tag_idxs, token_idxs in zip(tag_idxs_batch, token_idxs_batch):
        tags, tokens = [], []
        for tag_idx, token_idx in zip(tag_idxs, token_idxs):
            tags.append(idx2tag[tag_idx])
            tokens.append(idx2token[token_idx])
        tags_batch.append(tags)
        tokens_batch.append(tokens)
    return tags_batch, tokens_batch
    
    
def eval_conll(model, session, tokens, tags, short_report=True):
    
    y_true, y_pred = [], []
    for x_batch, y_batch, lengths in batches_generator(1, tokens, tags):
        tags_batch, tokens_batch = predict_tags(model, session, x_batch, lengths)
        if len(x_batch[0]) != len(tags_batch[0]):
            raise Exception("Incorrect length of prediction for the input, "
                            "expected length: %i, got: %i" % (len(x_batch[0]), len(tags_batch[0])))
        predicted_tags = []
        ground_truth_tags = []
        for gt_tag_idx, pred_tag, token in zip(y_batch[0], tags_batch[0], tokens_batch[0]): 
            if token != '<PAD>':
                ground_truth_tags.append(idx2tag[gt_tag_idx])
                predicted_tags.append(pred_tag)
        y_true.extend(ground_truth_tags + ['O'])
        y_pred.extend(predicted_tags + ['O'])
        
    results = precision_recall_f1(y_true, y_pred, print_results=True, short_report=short_report)
    return results

tf.reset_default_graph()

model =BiLSTMModel(53060,7,200,200,0)

batch_size =32
n_epochs = 4
learning_rate =  0.005
learning_rate_decay = 2
dropout_keep_probability = 0.9
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Start training... \n')
for epoch in range(n_epochs):
    print('-' * 20 + ' Epoch {} '.format(epoch+1) + 'of {} '.format(n_epochs) + '-' * 20)
    print('Train data evaluation:')
    eval_conll(model, sess, train_tokens, train_tags, short_report=True)
    print('Validation data evaluation:')
    eval_conll(model, sess, validation_tokens, validation_tags, short_report=True)
    
    for x_batch, y_batch, lengths in batches_generator(batch_size, train_tokens, train_tags):
        model.train_on_batch(sess, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability)
        
    learning_rate = learning_rate / learning_rate_decay
    
print('...training finished.')

print('-' * 20 + ' Train set quality: ' + '-' * 20)
train_results = eval_conll(model, sess, train_tokens, train_tags, short_report=False)

print('-' * 20 + ' Validation set quality: ' + '-' * 20)
validation_results = eval_conll(model, sess, validation_tokens, validation_tags, short_report=False)

print('-' * 20 + ' Test set quality: ' + '-' * 20)
test_results = eval_conll(model, sess, test_tokens, test_tags, short_report=False)

