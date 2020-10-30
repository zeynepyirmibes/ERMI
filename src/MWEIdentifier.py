# -*- coding: utf-8 -*-

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.layers import *
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint
import os

class MWEIdentifier:
    def __init__(self, language, embedding_type, mwe, logger, mwe_write_path):
        self.logger = logger
        self.logger.info('Initialize MWEIdentifier for %s' % language)
        self.language = language
        self.mwe = mwe
        self.embed = embedding_type
        self.write_path = mwe_write_path

    def set_params(self, params):
        self.logger.info('Setting params...')
        self.n_units = params['n_units']
        self._dropout = params['dropout']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']

    def head_word(self, word, head_id, sent):
        head = ''
        if head_id.strip() == '0':
            head = "<UNK>"
        else:
            for w in sent:
                if w[0] == head_id:
                    if w[1] == '_':
                        #print("word: ", word, " , head_word_lemma: ", w[2])
                        head = self.head_word(w[1], w[6], sent)
                    else:
                        #print("word: ", word, " , head_word: ", w[1])
                        head = w[1]
        if head == '':
            #print("word: ", word, " , head_id: ", head_id)
            head = "<UNK>"
        return head

    def set_test(self):
        self.logger.info('Setting test environment...')
        self.X_tr_pos = [[self.mwe.pos2idx[w[3]] for w in s] for s in self.mwe.train_sentences]
        self.X_tr_pos = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_tr_pos, padding="post", value=0)
        self.X_te_pos = [[self.mwe.pos2idx[w[3]] for w in s] for s in self.mwe.test_sentences]
        self.X_te_pos = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_te_pos, padding="post", value=0)

        self.X_tr_deprel = [[self.mwe.deprel2idx[w[7]] for w in s] for s in self.mwe.train_sentences]
        self.X_tr_deprel = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_tr_deprel, padding="post", value=0)
        self.X_te_deprel = [[self.mwe.deprel2idx[w[7]] for w in s] for s in self.mwe.test_sentences]
        self.X_te_deprel = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_te_deprel, padding="post", value=0)

        self.X_tr_word = [[self.mwe.word2idx[w[1]] for w in s] for s in self.mwe.train_sentences]
        self.X_tr_word = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_tr_word, padding="post", value=0)
        self.X_te_word = [[self.mwe.word2idx[w[1]] for w in s] for s in self.mwe.test_sentences]
        self.X_te_word = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_te_word, padding="post", value=0)

        if self.embed in ('head'):
            self.X_tr_head = [[self.mwe.word2idx[self.head_word(w[1], w[6], s)] for w in s] for s in self.mwe.train_sentences]  # head = 6
            self.X_tr_head = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_tr_head, padding="post", value=0)
            self.X_te_head = [[self.mwe.word2idx[self.head_word(w[1], w[6], s)] for w in s] for s in self.mwe.test_sentences]
            self.X_te_head = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_te_head, padding="post", value=0)

        self.y = [[self.mwe.tag2idx[w[-1]] for w in s] for s in self.mwe.train_sentences]
        self.y = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.y, padding="post", value=self.mwe.tag2idx["O"])
        self.y = [to_categorical(i, num_classes=self.mwe.n_tags) for i in self.y]

    def get_raw_set(self, raw_sentences):
        X_raw_pos = [[self.mwe.pos2idx[w[3]] for w in s] for s in raw_sentences]
        X_raw_pos = pad_sequences(maxlen=self.mwe.max_sent, sequences=X_raw_pos, padding="post", value=0)
        X_raw_deprel = [[self.mwe.deprel2idx[w[7]] for w in s] for s in raw_sentences]
        X_raw_deprel = pad_sequences(maxlen=self.mwe.max_sent, sequences=X_raw_deprel, padding="post", value=0)
        X_raw_word = [[self.mwe.word2idx[w[1]] for w in s] for s in raw_sentences]
        X_raw_word = pad_sequences(maxlen=self.mwe.max_sent, sequences=X_raw_word, padding="post", value=0)

        X_raw_head = [[self.mwe.word2idx[self.head_word(w[1], w[6], s)] for w in s] for s in raw_sentences]  # head = 6
        X_raw_head = pad_sequences(maxlen=self.mwe.max_sent, sequences=X_raw_head, padding="post", value=0)
        """self.y = [[self.mwe.tag2idx[w[-1]] for w in s] for s in self.mwe.train_sentences]
        self.y = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.y, padding="post", value=self.mwe.tag2idx["O"])
        self.y = [to_categorical(i, num_classes=self.mwe.n_tags) for i in self.y]"""
        return X_raw_pos, X_raw_deprel, X_raw_word, X_raw_head

    def build_model(self):
        self.build_model_with_pretrained_embedding()

    def build_model_with_pretrained_embedding(self):
        self.logger.info('Building model with pretrained embedding...')
        tokens_input = Input(shape=(None,), name='words_input')
        tokens = Embedding(input_dim=self.mwe.word_embeddings.shape[0], output_dim=self.mwe.word_embeddings.shape[1],
                           weights=[self.mwe.word_embeddings],
                           trainable=False, mask_zero=True, input_length=self.mwe.max_sent, name='word_embeddings')(
            tokens_input)

        if self.embed in ('head'):
            head_input = Input(shape=(None,), name='headwords_input')
            heads = Embedding(input_dim=self.mwe.word_embeddings.shape[0], output_dim=self.mwe.word_embeddings.shape[1],
                               weights=[self.mwe.word_embeddings],
                               trainable=False, mask_zero=True, input_length=self.mwe.max_sent, name='head_embeddings')(
                head_input)

        pos_embedding = np.identity(len(self.mwe.pos2idx.keys()) + 1)
        pos_input = Input(shape=(None,), name='pos_input')
        pos_tokens = Embedding(input_dim=pos_embedding.shape[0], output_dim=pos_embedding.shape[1],
                               weights=[pos_embedding],
                               trainable=False, mask_zero=True, input_length=self.mwe.max_sent,
                               name='pos_embeddings')(
            pos_input)

        deprel_embedding = np.identity(len(self.mwe.deprel2idx.keys()) + 1)
        deprel_input = Input(shape=(None,), name='deprel_input')
        deprel_tokens = Embedding(input_dim=deprel_embedding.shape[0], output_dim=deprel_embedding.shape[1],
                                  weights=[deprel_embedding],
                                  trainable=False, mask_zero=True, input_length=self.mwe.max_sent,
                                  name='deprel_embeddings')(
            deprel_input)

        if self.embed == 'head':
            print("Headword embeddings added to the end.")
            inputNodes = [tokens_input, pos_input, deprel_input, head_input]
            mergeInputLayers = [tokens, pos_tokens, deprel_tokens, heads]
        else:
            print("No headword embeddings added.")
            inputNodes = [tokens_input, pos_input, deprel_input]
            mergeInputLayers = [tokens, pos_tokens, deprel_tokens]
        merged_input = concatenate(mergeInputLayers)

        shared_layer = merged_input
        shared_layer = Bidirectional(LSTM(self.n_units, return_sequences=True, dropout=self._dropout[0],
                                          recurrent_dropout=self._dropout[1]),
                                     name='shared_varLSTM')(shared_layer)

        output = shared_layer
        output = TimeDistributed(Dense(self.mwe.n_tags, activation=None))(output)
        crf = CRF(self.mwe.n_tags)  # CRF layer
        output = crf(output)  # output

        model = Model(inputs=inputNodes, outputs=[output])
        model.compile(optimizer="nadam", loss=crf.loss_function, metrics=[crf.accuracy])
        self.model = model

    def fit_model(self):
        self.logger.info('Fitting model...')
        # checkpoint
        """loss_filepath= os.path.join(self.write_path, "teacher-weights-least.hdf5")
        acc_filepath= os.path.join(self.write_path, "teacher-weights-best.hdf5")
        loss_checkpoint = ModelCheckpoint(loss_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        acc_checkpoint = ModelCheckpoint(acc_filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [loss_checkpoint, acc_checkpoint]"""
        # Fit the model
        if self.embed == 'head':
            self.model.fit(
                {'words_input': self.X_tr_word, 'pos_input': self.X_tr_pos, 'deprel_input': self.X_tr_deprel, 'headwords_input': self.X_tr_head},
                np.array(self.y),
                batch_size=self.batch_size, epochs=self.epochs, verbose=1)
        else:
            self.model.fit(
                {'words_input': self.X_tr_word, 'pos_input': self.X_tr_pos, 'deprel_input': self.X_tr_deprel},
                np.array(self.y),
                batch_size=self.batch_size, epochs=self.epochs, verbose=1)
        last_filepath= os.path.join(self.write_path, "teacher-weights-last.hdf5")
        self.model.save_weights(last_filepath)
        
    def predict(self):
        self.logger.info('Predicting...')
        predicted_tags = []
        if self.embed == 'head':
            preds = self.model.predict([self.X_te_word, self.X_te_pos, self.X_te_deprel, self.X_te_head])
        else:
            preds = self.model.predict([self.X_te_word, self.X_te_pos, self.X_te_deprel])
        for i in range(self.X_te_word.shape[0]):
            p = preds[i]
            p = np.argmax(p, axis=-1)
            tp = []
            for w, pred in zip(self.X_te_word[i], p):
                if w != 0:
                    tp.append(self.mwe.tags[pred])
            predicted_tags.append(tp)
        self.predicted_tags = predicted_tags

    def add_tags_to_test(self):
        self.logger.info('Tagging...')
        tags = []
        for i in range(len(self.predicted_tags)):
            for j in range(len(self.predicted_tags[i])):
                tags.append(self.predicted_tags[i][j])
            tags.append('space')
        self.mwe._test_corpus['BIO'] = tags
        self.mwe.convert_tag()
        
    def predict_test_custom_model(self, reload_path):
        if os.path.exists(reload_path):
            self.model.load_weights(reload_path)
            test_extension = reload_path.replace(os.path.join(self.write_path,'teacher-weights-'), '').replace('.hdf5', '')
            test_filename = 'test_tagged_' + test_extension + '.cupt'
            self.mwe.test_tagged_path = os.path.join(self.write_path, test_filename)
            self.logger.info('Predicting for ', self.mwe.test_tagged_path , "with ", reload_path)
            self.predict()
            self.logger.info('add_tags_to_test...')
            self.add_tags_to_test()
        
    def predict_raw_custom_model(self, reload_path):
        if os.path.exists(reload_path):
            self.model.load_weights(reload_path)
            raw_extension = reload_path.replace(os.path.join(self.write_path,'teacher-weights-'), '').replace('.hdf5', '')
            test_filename = 'test_tagged_' + test_extension + '.cupt'
            self.mwe.test_tagged_path = os.path.join(self.write_path, test_filename)
            self.logger.info('Predicting for ', self.mwe.test_tagged_path , "with ", reload_path)
            self.predict()
            self.logger.info('add_tags_to_test...')
            self.add_tags_to_test()
