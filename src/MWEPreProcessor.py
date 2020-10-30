# -*- coding: utf-8 -*-

import re
import copy
import io
import os
import codecs
import pandas as pd


class MWEPreProcessor:
    def __init__(self, language, withraw, root_path, write_path):
        print('İnitialize MWEPreprocessor for %s' % language)
        self.language = language
        self.root_path = os.path.join(root_path, language)
        self.write_path = write_path
        self.withraw = withraw
        self.set_paths()
        
    def set_tagging(self, tagging):
        self.tagging = tagging

    def update_root_path(self, root_path):
        self.root_path = root_path + '/%s/' % self.language
        self.set_paths()

    def set_paths(self):
        self.train_path = os.path.join(self.root_path, 'train.cupt')
        self.dev_path = os.path.join(self.root_path, 'dev.cupt')
        self.train_pkl_path = os.path.join(self.root_path, 'train.pkl')
        self.train_tagged_path = os.path.join(self.write_path, 'train_tagged.cupt')
        self.test_pkl_path = os.path.join(self.root_path, 'test.pkl')
        self.test_blind_path = os.path.join(self.root_path, 'test.blind.cupt')
        self.test_gold_path = os.path.join(self.root_path, 'test.cupt')
        self.test_tagged_path = os.path.join(self.write_path, 'test_tagged.cupt')
        self.raw_path = os.path.join(self.write_path, 'raw_tagged.cupt')

        self.model_pkl_path = os.path.join(self.root_path, 'model.pkl')

    def read_corpus(self, path):
        corpus_file = io.open(path, "r", encoding="utf-8")
        corpus = []
        for s in corpus_file:
            if not s.startswith('#'):
                corpus.append(s)
        corpus = [x.split('\t') for x in corpus]
        new_corpus = pd.DataFrame(corpus,
                                  columns=['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL',
                                           'DEPS', 'MISC',
                                           'PARSEME:MWE'])
        return new_corpus

    def to_cupt(self, df, cupt_path):
        print('Writing to %s...' % cupt_path)
        lines = ''
        for idx, row in df.iterrows():
            if row['PARSEME:MWE'] == 'space':
                line = '\n'
            else:
                line = str(row['ID']) + '\t' + row['FORM'] + '\t' + row['LEMMA'] + '\t' + row['UPOS'] + '\t' + row[
                    'XPOS'] + '\t' + row['FEATS'] + '\t' + row['HEAD'] + '\t' + row['DEPREL'] + '\t' + row[
                           'DEPS'] + '\t' + row['MISC'] + '\t' + row['BIO'] + '\n'
            lines += line

        f = codecs.open(cupt_path, "w", "utf-8")
        f.write(lines)
        f.close()

    def find_comments_in_cupt(self, path):
        cupt_file = io.open(path, "r", encoding="utf-8")
        comments = []
        isSpaceAdded = False
        for s in cupt_file:
            if s.startswith('#'):
                comments.append(s)
                isSpaceAdded = False
            elif not isSpaceAdded:
                comments.append(" ")
                isSpaceAdded = True
        return comments

    def to_cupt_with_comments(self):
        comments = self.find_comments_in_cupt(self.test_blind_path)
        print('Writing to %s...' % self.test_tagged_path)
        lines = ''
        counterC = 0
        while not comments[counterC] == " ":
            line = comments[counterC]
            lines += line
            counterC = counterC + 1
        counterC = counterC + 1
        for idx, row in self._test_corpus.iterrows():
            if row['PARSEME:MWE'] == 'space':
                line = '\n'
                while counterC < len(comments) and not comments[counterC] == " ":
                    line = line + comments[counterC]
                    counterC = counterC + 1
                if comments[counterC] == " ":
                    if counterC + 1 < len(comments):
                        counterC = counterC + 1
            else:
                line = str(row['ID']) + '\t' + row['FORM'] + '\t' + row['LEMMA'] + '\t' + row['UPOS'] + '\t' + row[
                    'XPOS'] + '\t' + row['FEATS'] + '\t' + row['HEAD'] + '\t' + row['DEPREL'] + '\t' + row[
                           'DEPS'] + '\t' + row['MISC'] + '\t' + row['BIO'] + '\n'
            lines += line
        f = codecs.open(self.test_tagged_path, "w", "utf-8")
        f.write(lines)
        f.close()

    def read_sentences(self, corpus):
        sentence_indexes = [-1] + list(corpus.loc[corpus['BIO'] == 'space'].index)
        sentences = []
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = corpus[sentence_idx + 1:sentence_indexes[i + 1]]
            tr_sentence = []
            for j, row in sentence.iterrows():
                tr_sentence.append((row['ID'], row['FORM'], row['LEMMA'], row['UPOS'], row['XPOS'], row['FEATS'],
                                    row['HEAD'], row['DEPREL'], row['DEPS'], row['MISC'], row['BIO']))
            sentences.append(tr_sentence)
        return sentences
    
    def read_raw_sentences(self, corpus):
        sentence_indexes = [-1] + list(corpus.loc[corpus['BIO'] == 'space'].index)
        sentences = []
        raw_count = 0
        max_train_sent = max([len(sen) for sen in self.train_sentences])
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = corpus[sentence_idx + 1:sentence_indexes[i + 1]]
            tr_sentence = []
            for j, row in sentence.iterrows():
                tr_sentence.append((row['ID'], row['FORM'], row['LEMMA'], row['UPOS'], row['XPOS'], row['FEATS'],
                                    row['HEAD'], row['DEPREL'], row['DEPS'], row['MISC'], row['BIO']))
            if len(tr_sentence) <= max_train_sent and raw_count <= len(self.train_sentences):
                sentences.append(tr_sentence)
                raw_count += 1
        print("Raw sentences read: ", len(sentences))
        return sentences

    def update_test_corpus(self):
        self._test_corpus['BIO'] = copy.deepcopy(self._test_corpus['PARSEME:MWE'])
        self._test_corpus[self._test_corpus['BIO'].isnull()] = 'space'
        self._test_corpus['BIO'] = copy.deepcopy(self._test_corpus['BIO'].apply(lambda x: x.strip()))
        return self._test_corpus

    def set_train_dev(self):
        train_corpus = self.read_corpus(self.train_path)
        if not os.path.isfile(self.dev_path):
            print('Dev set is not found.')
            dev_corpus = pd.DataFrame()
        else:
            dev_corpus = self.read_corpus(self.dev_path)
            
        if self.withraw == 'yes':
            if not os.path.isfile(self.raw_path):
                print('Raw set is not found, Train = train + dev')
                raw_corpus = pd.DataFrame()
                corpus = pd.concat([train_corpus, dev_corpus, raw_corpus], ignore_index=True)
                self._train_corpus = corpus
            else:
                print('Raw set is found, Train = raw')
                raw_corpus = self.read_corpus(self.raw_path)
                self._raw_corpus = raw_corpus
                self._train_corpus = raw_corpus
        else:
            print('Train = train + dev')
            corpus = pd.concat([train_corpus, dev_corpus], ignore_index=True)
            self._train_corpus = corpus

    def set_raw_corpus(self, corpus):
        self._raw_corpus = corpus

    def set_test_corpus(self):
        self._test_corpus = self.read_corpus(self.test_blind_path)

    def set_word_embeddings(self, word_embeddings):
        self.word_embeddings = word_embeddings

    def tag(self):
        if self.tagging == 'IOB':
            self.tag_IOB()
        elif self.tagging == 'gappy-1':
            self.tag_gappy_1_level()
        elif self.tagging == 'gappy-crossy':
            self.tag_bigappy_unicrossy()

    def convert_tag(self):
        if self.tagging == 'IOB':
            self.convert_IOB()
        elif self.tagging == 'gappy-1':
            self.convert_gappy_tag()
        elif self.tagging == 'gappy-crossy':
            self.convert_gappy_tag()

    def tag_bigappy_unicrossy(self):
        self._train_corpus['BIO'] = copy.deepcopy(self._train_corpus['PARSEME:MWE'])
        self._train_corpus[self._train_corpus['BIO'].isnull()] = 'space'

        # remove other tags after the first tag
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: x.split(';')[0] if bool(re.match("\d:\w+[.]*\w+;.", x)) else x)
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: x.split(';')[0] if bool(re.match("\d;.", x)) else x)
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(lambda x: x.strip())

        sentence_indexes = [-1] + list(self._train_corpus.loc[self._train_corpus['BIO'] == 'space'].index)

        # tag sentence by sentence
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = self._train_corpus[sentence_idx + 1:sentence_indexes[i + 1]]
            o_indexes = []
            last_B_idx = 0
            last_b_idx = 0
            isB = False
            # j -> each token in the sentence
            # only 1 B(I) and 1 b(i) is taken into consideration simultaneously
            # allows crossings
            for j, row in sentence.iterrows():
                tag = sentence.loc[j, 'BIO']
                # if (there is a tag in the form of no:category and it is the first VMWE in the sentence)
                # or (there is a tag in the form of no:category
                # and it is the beginning of an VMWE after the last B(I) tagged VMWE ends)
                # it does not wait the end of the nested VMWE to begin a new B(I) tagged VMWE
                # B I b i I I B i I is possible
                # only 1 B is allowed - 1 level
                if (bool(re.match("\d:", tag)) and not isB) or (bool(re.match("\d:", tag)) and j > last_B_idx):
                    isB = True
                    no = tag.split(':')[0]
                    category = tag.split(':')[1]
                    category = category.strip()
                    self._train_corpus.loc[j, 'BIO'] = 'B:' + category
                    # stores I indexes
                    I_indexes = list(sentence.loc[self._train_corpus['BIO'] == no].index)
                    # if it is multi-token VMWE
                    if len(I_indexes) > 0:
                        for k in I_indexes:
                            self._train_corpus.loc[k, 'BIO'] = 'I:' + category
                        last_B_idx = I_indexes[-1]
                        # it is multi-token VMWE, so there is a possibility of gap
                        # add the beginning index and the end index of the VMWE
                        o_indexes.append([j, last_B_idx])
                    # if it is single-token VMWE
                    else:
                        last_B_idx = j

                # if (there is a tag in the form of no:category and
                # it is a nested VMWE since its in between the last BI tagged VMWE)
                # and it is the first nested VMWE in between the last BI tagged VMWE
                # only 1 b is allowed - 1 level
                # since the location of i is not checked, it allows crossing
                elif (bool(re.match("\d:", tag)) and j < last_B_idx) and j > last_b_idx:
                    no = tag.split(':')[0]
                    category = tag.split(':')[1]
                    category = category.strip()
                    self._train_corpus.loc[j, 'BIO'] = 'b:' + category
                    # stores i indexes
                    i_indexes = list(sentence.loc[self._train_corpus['BIO'] == no].index)
                    # if it is multi-token VMWE
                    if len(i_indexes) > 0:
                        for l in i_indexes:
                            self._train_corpus.loc[l, 'BIO'] = 'i:' + category
                        last_b_idx = i_indexes[-1]
                        # it is multi-token VMWE, so there is a possibility of gap
                        # add the beginning index and the end index of the VMWE
                        o_indexes.append([j, last_b_idx])
                    # if it is single-token VMWE
                    else:
                        last_b_idx = j

            for o_idx in o_indexes:
                for oo_idx in range(o_idx[0] + 1, o_idx[1]):
                    tag = sentence.loc[oo_idx, 'BIO']
                    # if there is no tag in between an multi-token MWE, tag with 'o'
                    if not (bool(re.match("I:", tag)) or bool(re.match("B:", tag)) or bool(re.match("i:", tag)) or bool(
                            re.match("b:", tag))):
                        self._train_corpus.loc[oo_idx, 'BIO'] = 'o'

        # tags with 'O'
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: 'O' if not (bool(re.match("i:", x)) or bool(re.match("b:", x)) or bool(re.match("I:", x)) or bool(
                re.match("B:", x)) or bool(re.match("o", x)) or bool(re.match("space", x))) else x)

        # self._train_corpus.to_csv(fileName)
        self.to_cupt(self._train_corpus, self.train_tagged_path)

    def tag_gappy_1_level(self):
        self._train_corpus['BIO'] = copy.deepcopy(self._train_corpus['PARSEME:MWE'])
        self._train_corpus[self._train_corpus['BIO'].isnull()] = 'space'

        # remove other tags after the first tag
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: x.split(';')[0] if bool(re.match("\d:\w+[.]*\w+;.", x)) else x)
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: x.split(';')[0] if bool(re.match("\d;.", x)) else x)
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(lambda x: x.strip())

        sentence_indexes = [-1] + list(self._train_corpus.loc[self._train_corpus['BIO'] == 'space'].index)

        # tag sentence by sentence
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = self._train_corpus[sentence_idx + 1:sentence_indexes[i + 1]]
            o_indexes = []
            last_B_idx = 0
            last_b_idx = 0
            isB = False
            # j -> each token in the sentence
            # only 1 B(I) and 1 b(i) is taken into consideration simultaneously
            # does not allow crossings
            for j, row in sentence.iterrows():
                tag = sentence.loc[j, 'BIO']
                # if (there is a tag in the form of no:category and it is the first VMWE in the sentence)
                # or (there is a tag in the form of no:category
                # and it is the beginning of an VMWE after the last B(I) tagged VMWE ends)
                # only 1 B is allowed - 1 level
                if (bool(re.match("\d:", tag)) and not isB) or (bool(re.match("\d:", tag)) and j > last_B_idx):
                    isB = True
                    no = tag.split(':')[0]
                    category = tag.split(':')[1]
                    category = category.strip()
                    self._train_corpus.loc[j, 'BIO'] = 'B:' + category
                    # stores I indexes
                    I_indexes = list(sentence.loc[self._train_corpus['BIO'] == no].index)
                    # if it is multi-token VMWE
                    if len(I_indexes) > 0:
                        for k in I_indexes:
                            self._train_corpus.loc[k, 'BIO'] = 'I:' + category
                        last_B_idx = I_indexes[-1]
                        # it is multi-token VMWE, so there is a possibility of gap
                        # add the beginning index and the end index of the VMWE
                        o_indexes.append([j, last_B_idx])
                    # if it is single-token VMWE
                    else:
                        last_B_idx = j

                # if (there is a tag in the form of no:category and
                # it is a nested VMWE since its in between the last BI tagged VMWE)
                # and it is the first nested VMWE in between the last BI tagged VMWE
                # only 1 b is allowed - 1 level
                # since the location of i is checked, it does not allow crossing
                elif (bool(re.match("\d:", tag)) and j < last_B_idx) and j > last_b_idx:
                    prev_last_b_idx = last_b_idx
                    no = tag.split(':')[0]
                    category = tag.split(':')[1]
                    category = category.strip()
                    # stores i indexes
                    i_indexes = list(sentence.loc[self._train_corpus['BIO'] == no].index)
                    # if it is multi-token VMWE
                    if len(i_indexes) > 0:
                        valid_b = 0
                        last_b_idx = i_indexes[-1]
                        if last_b_idx < last_B_idx:
                            valid_b = 1
                            if not i_indexes[0] - j == 1:
                                valid_b = 0
                            for i_idx in range(1, len(i_indexes)):
                                if not i_indexes[i_idx] - i_indexes[i_idx - 1] == 1:
                                    valid_b = 0
                            if valid_b == 1:
                                for n_idx in range(j + 1, last_b_idx):
                                    tagg = sentence.loc[n_idx, 'BIO']
                                    # if there is no tag in between an nested MWE, tag with nested MWE
                                    # if there is a tag, invalid nested MWE
                                    if bool(re.match("I:", tagg)) or bool(re.match("B:", tagg)) or bool(
                                            re.match("i:", tagg)) or bool(re.match("b:", tagg)):
                                        valid_b = 0
                        if valid_b == 1:
                            self._train_corpus.loc[j, 'BIO'] = 'b:' + category
                            for l in i_indexes:
                                self._train_corpus.loc[l, 'BIO'] = 'i:' + category
                        if valid_b == 0:
                            last_b_idx = prev_last_b_idx
                    # if it is single-token VMWE
                    else:
                        self._train_corpus.loc[j, 'BIO'] = 'b:' + category
                        last_b_idx = j

            for o_idx in o_indexes:
                for oo_idx in range(o_idx[0] + 1, o_idx[1]):
                    tag = sentence.loc[oo_idx, 'BIO']
                    # if there is no tag in between an multi-token MWE, tag with 'o'
                    if not (bool(re.match("I:", tag)) or bool(re.match("B:", tag)) or bool(re.match("i:", tag)) or bool(
                            re.match("b:", tag))):
                        self._train_corpus.loc[oo_idx, 'BIO'] = 'o'

        # tags with 'O'
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: 'O' if not (bool(re.match("i:", x)) or bool(re.match("b:", x)) or bool(re.match("I:", x)) or bool(
                re.match("B:", x)) or bool(re.match("o", x)) or bool(re.match("space", x))) else x)

        # self._train_corpus.to_csv(fileName)
        self.to_cupt(self._train_corpus, self.train_tagged_path)

    def tag_IOB(self):
        self._train_corpus['BIO'] = copy.deepcopy(self._train_corpus['PARSEME:MWE'])
        self._train_corpus[self._train_corpus['BIO'].isnull()] = 'space'

        # remove other tags after the first tag
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: x.split(';')[0] if bool(re.match("\d:\w+[.]*\w+;.", x)) else x)
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: x.split(';')[0] if bool(re.match("\d;.", x)) else x)
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(lambda x: x.strip())

        sentence_indexes = [-1] + list(self._train_corpus.loc[self._train_corpus['BIO'] == 'space'].index)

        # tag sentence by sentence
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = self._train_corpus[sentence_idx + 1:sentence_indexes[i + 1]]
            last_B_idx = 0
            isB = False
            for j, row in sentence.iterrows():
                tag = sentence.loc[j, 'BIO']
                if (bool(re.match("\d:", tag)) and not isB) or (bool(re.match("\d:", tag)) and j > last_B_idx):
                    isB = True
                    no = tag.split(':')[0]
                    category = tag.split(':')[1]
                    category = category.strip()
                    self._train_corpus.loc[j, 'BIO'] = 'B:' + category
                    I_indexes = list(sentence.loc[self._train_corpus['BIO'] == no].index)
                    if len(I_indexes) > 0:
                        for k in I_indexes:
                            self._train_corpus.loc[k, 'BIO'] = 'I:' + category
                        last_B_idx = I_indexes[-1]
                    else:
                        last_B_idx = j

        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: 'O' if not (
                    bool(re.match("I:", x)) or bool(re.match("B:", x)) or bool(re.match("space", x))) else x)

        # self._train_corpus.to_csv(fileName)
        self.to_cupt(self._train_corpus, self.train_tagged_path)

    def convert_gappy_tag(self):
        self._test_corpus['PARSEME:MWE'] = copy.deepcopy(self._test_corpus['BIO'])
        sentence_indexes = [-1] + list(self._test_corpus.loc[self._test_corpus['BIO'] == 'space'].index)

        # tag sentence by sentence
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = self._test_corpus[sentence_idx + 1:sentence_indexes[i + 1]]
            counter = 0
            counterB = 0
            tagB = ""
            tagb = ""
            counterb = 0
            for j, row in sentence.iterrows():
                tag = sentence.loc[j, 'BIO']
                # tag = tag.strip() ###### ekle for FR and PL if necessary

                if tag == 'O':
                    self._test_corpus.loc[j, 'BIO'] = '*'

                elif tag == 'o':
                    self._test_corpus.loc[j, 'BIO'] = '*'

                elif not tag == 'space':
                    ib = tag.split(':')[0]
                    category = tag.split(':')[1]

                    if ib == "B":
                        counter = counter + 1
                        counterB = counter
                        tagB = category
                        self._test_corpus.loc[j, 'BIO'] = str(counterB) + ':' + tagB

                    if ib == "b":
                        counter = counter + 1
                        counterb = counter
                        tagb = category
                        self._test_corpus.loc[j, 'BIO'] = str(counterb) + ':' + tagb

                    if ib == "I" and tagB == category:
                        self._test_corpus.loc[j, 'BIO'] = str(counterB)

                    if ib == "i" and tagb == category:
                        self._test_corpus.loc[j, 'BIO'] = str(counterb)

                    if ib == "I" and (not (tagB == category)):
                        counter = counter + 1
                        counterB = counter
                        tagB = category
                        self._test_corpus.loc[j, 'BIO'] = str(counterB) + ':' + tagB

                    if ib == "i" and (not (tagb == category)):
                        counter = counter + 1
                        counterb = counter
                        tagb = category
                        self._test_corpus.loc[j, 'BIO'] = str(counterb) + ':' + tagb

        self.to_cupt_with_comments()

    def convert_IOB(self):
        self._test_corpus['PARSEME:MWE'] = copy.deepcopy(self._test_corpus['BIO'])
        sentence_indexes = [-1] + list(self._test_corpus.loc[self._test_corpus['BIO'] == 'space'].index)

        # tag sentence by sentence
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = self._test_corpus[sentence_idx + 1:sentence_indexes[i + 1]]
            counter = 0
            counterB = 0
            tagB = ""
            for j, row in sentence.iterrows():
                tag = sentence.loc[j, 'BIO']
                # tag = tag.strip() ###### ekle for FR and PL if necessary

                if tag == 'O':
                    self._test_corpus.loc[j, 'BIO'] = '*'

                elif not tag == 'space':
                    ib = tag.split(':')[0]
                    category = tag.split(':')[1]

                    if ib == "B":
                        counter = counter + 1
                        counterB = counter
                        tagB = category
                        self._test_corpus.loc[j, 'BIO'] = str(counterB) + ':' + tagB

                    if ib == "I" and tagB == category:
                        self._test_corpus.loc[j, 'BIO'] = str(counterB)

                    if ib == "I" and (not (tagB == category)):
                        counter = counter + 1
                        counterB = counter
                        tagB = category
                        self._test_corpus.loc[j, 'BIO'] = str(counterB) + ':' + tagB

        self.to_cupt_with_comments()

    def prepare_to_lstm(self):
        print('Preparing to lstm..')
        self.train_sentences = self.read_sentences(self._train_corpus)
        self.test_sentences = self.read_sentences(self._test_corpus)
        if self.withraw == 'yes' or os.path.isfile(self.raw_path):
            self.raw_sentences = self.read_raw_sentences(self._raw_corpus)

        self.words = []
        self.words.append("</s>")
        self.words.append("<UNK>")
        if self.withraw == 'yes' or os.path.isfile(self.raw_path):
            self.words = self.words + list(set(self._train_corpus['FORM']) | set(self._test_corpus['FORM']) | set(self._raw_corpus['FORM']))
        else:
            self.words = self.words + list(set(self._train_corpus['FORM']) | set(self._test_corpus['FORM']))
        self.tags = list(set(self._train_corpus['BIO']))
        self.tags.remove("space")

        self.pos = []
        self.pos.append("</s>")
        if self.withraw == 'yes' or os.path.isfile(self.raw_path):
            self.pos = self.pos + list(set(self._train_corpus['UPOS']) | set(self._test_corpus['UPOS']) | set(self._raw_corpus['UPOS']))
        else:
            self.pos = self.pos + list(set(self._train_corpus['UPOS']) | set(self._test_corpus['UPOS']))
        self.pos.remove("space")

        self.deprel = []
        self.deprel.append("</s>")
        if self.withraw == 'yes' or os.path.isfile(self.raw_path):
            self.deprel = self.deprel + list(set(self._train_corpus['DEPREL']) | set(self._test_corpus['DEPREL']) | set(self._raw_corpus['DEPREL']))
        else:
            self.deprel = self.deprel + list(set(self._train_corpus['DEPREL']) | set(self._test_corpus['DEPREL']))
        self.deprel.remove("space")

        self.n_words = len(self.words)
        self.n_tags = len(self.tags)
        self.word2idx = {w: i for i, w in enumerate(self.words)}
        self.pos2idx = {t: i for i, t in enumerate(self.pos)}
        self.tag2idx = {t: i for i, t in enumerate(self.tags)}
        self.deprel2idx = {t: i for i, t in enumerate(self.deprel)}
        self.max_train_sent = max([len(sen) for sen in self.train_sentences])
        self.max_test_sent = max([len(sen) for sen in self.test_sentences])
        self.max_sent = max(self.max_train_sent, self.max_test_sent)
        if self.withraw == 'yes' or os.path.isfile(self.raw_path):
            self.max_raw_sent = max([len(sen) for sen in self.raw_sentences])
            
