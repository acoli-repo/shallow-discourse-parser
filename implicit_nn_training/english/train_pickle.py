# -*- coding: utf-8 -*-
import json
import gensim
import logging
import climate
import theanets
import numpy as np
import re
import collections
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import random
import cPickle
import os

def start_vectors(counter, parses_train_filepath, parses_dev_filepath, relations_train_filepath, relations_dev_filepath, 
                     googlevecs_filepath):
    """ train vectors """
    mean = (lambda x: sum(x)/float(len(x)))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # Initalize semantic model (with None data)
    m = gensim.models.Word2Vec(None, size=300, window=8, min_count=3, workers=4, negative=5*0)
    #m = gensim.models.Doc2Vec(None, size=300, window=8, min_count=3, workers=4, negative=5*0)
    print "Reading data..."
    # Load parse file
    parses = json.load(open(parses_train_filepath))
    parses.update(json.load(open(parses_dev_filepath)))
    (relations_train, all_relations_train) = read_file(relations_train_filepath, parses)
    (relations_dev, all_relations_dev) = read_file(relations_dev_filepath, parses)
    relations = relations_train + relations_dev
    all_relations = all_relations_train + all_relations_dev
    # Substitution dictionary for class labels to integers
    label_subst = dict([(y,x) for x,y in enumerate(set([r[0][0] for r in relations]))])
    print("Label subst", label_subst)
    print "Build vocabulary..."
    m.build_vocab(RelReader(all_relations))
    #m.build_vocab(ParseReader(parses, docvec=True))
    print "Reading pre-trained word vectors..."
    m.intersect_word2vec_format(googlevecs_filepath, binary=True)
    print "Training segment vectors..."    
    for iter in range(1, 20):
        ## Training of word vectors
        m.alpha = 0.01/(2**iter)
        m.min_alpha = 0.01/(2**(iter+1))
        print "Vector training iter", iter, m.alpha, m.min_alpha
        m.train(ParseReader(parses))
    if not os.path.exists("pickles"):
        os.makedirs("pickles")
    file = open("pickles/m_"+str(counter)+".py", "wb")
    cPickle.dump(m, file, protocol=cPickle.HIGHEST_PROTOCOL)
    file.close()    
    (input_train, output_train) = convert_relations(relations_train, label_subst, m)
    (input_dev, output_dev) = convert_relations(relations_dev, label_subst, m)


    return input_train, output_train, input_dev, output_dev, label_subst


def train_theanet(counter_vec, counter_nn,method, learning_rate, momentum, decay, regularization, hidden,
                  min_improvement, validate_every, patience, weight_lx, hidden_lx,
                  input_train, output_train, input_dev, output_dev, label_subst):
    ''' train neural network, calculate confusion matrix, save neural network'''

    ## Training options:
    ## 1.) use 90% training set to train, use remaining 10% to validate, use test set to test
    #split_point = int(len(input_train)*0.9)
    #train_data = (input_train[:split_point], output_train[:split_point])
    #valid_data = (input_train[split_point:], output_train[split_point:])
    #test_data = (input_dev, output_dev)

    ## 2.) use 100% training set to train, use test set to validate and to test
    train_data = (input_train, output_train)
    test_data = (input_dev, output_dev)
    valid_data = (input_dev, output_dev)
    accs = []
    train_accs = []
    valid_accs = []
    
    exp = theanets.Experiment(theanets.Classifier, layers=(len(input_train[0]), hidden, len(label_subst)), loss='XE')
    if weight_lx == "l1":
        if hidden_lx == "l1":
            exp.train(train_data, valid_data, optimize=method,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            weight_l1=decay,
                                            hidden_l1=regularization,
                                            min_improvement=min_improvement, #0.02,
                                            validate_every=validate_every, #5,
                                            patience=patience)#=5
        else:
            exp.train(train_data, valid_data, optimize=method,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            weight_l1=decay,
                                            hidden_l2=regularization,
                                            min_improvement=min_improvement, #0.02,
                                            validate_every=validate_every, #5,
                                            patience=patience)#=5
    else:
        if hidden_lx == "l1":   
            exp.train(train_data, valid_data, optimize=method,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            weight_l2=decay,
                                            hidden_l1=regularization,
                                            min_improvement=min_improvement, #0.02,
                                            validate_every=validate_every, #5,
                                            patience=patience)#=5
        else:
            exp.train(train_data, valid_data, optimize=method,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            weight_l2=decay,
                                            hidden_l2=regularization,
                                            min_improvement=min_improvement, #0.02,
                                            validate_every=validate_every, #5,
                                            patience=patience)#=5          
    confmx = confusion_matrix(exp.network.predict(test_data[0]), test_data[1])
    acc = float(sum(np.diag(confmx)))/sum(sum(confmx))
    print "acc original: ", acc
    print "acc sklear: ", accuracy_score(exp.network.predict(test_data[0]), test_data[1])
    report = classification_report(exp.network.predict(test_data[0]),test_data[1])
    print classification_report(exp.network.predict(test_data[0]),test_data[1]), "\nAverage accuracy:", acc
    print "Confusion matrix:\n", confmx
    accs.append(acc)
    print "Mean accuracy", np.average(accs), np.std(accs)
    train_acc = accuracy_score(exp.network.predict(train_data[0]),train_data[1])
    train_accs.append(train_acc)
    print "Mean Train-Accuracy", np.average(train_accs), np.std(train_accs)
    valid_acc = accuracy_score(exp.network.predict(valid_data[0]),valid_data[1])
    valid_accs.append(valid_acc)
    file = open("pickles/neuralnetwork_"+str(counter_vec)+"_"+str(counter_nn)+"_.save", "wb")
    cPickle.dump(exp.network, file, protocol=cPickle.HIGHEST_PROTOCOL)
    file.close()
    file_ls = open("pickles/label_subst_"+str(counter_vec)+"_"+str(counter_nn)+"_.py", "wb")
    cPickle.dump(label_subst, file_ls, protocol=cPickle.HIGHEST_PROTOCOL)
    file_ls.close()
    return np.average(accs), np.average(valid_accs), np.average(train_accs)
   

class RelReader(object):
    """ Iterator for reading relation data """
    def __init__(self, segs, docvec=False):
        self.segs = segs
        self.docvec = docvec
    def __iter__(self):
        for file, i in zip(self.segs, range(len(self.segs))):
            for sub in [0, 1]:
                doclab = 'SEG_%d.%d' % (i, sub)
                #if i % 1000 == 0:
                    #print "      Reading", doclab
                text = [token for token, _ in self.segs[i][sub+1]]
                if self.docvec:
                    yield gensim.models.doc2vec.TaggedDocument(words=text, tags=i*2+sub)#[doclab])
                else:
                    yield text


class ParseReader(object):
    """ Iterator for reading parse data """
    def __init__(self, parses, docvec=False, offset=0):
        self.parses = parses
        self.docvec = docvec
        self.offset = offset
    def __iter__(self):
        i = -1
        for doc in self.parses:
            ####print "      Reading", doc
            for sent_i, sent in enumerate(self.parses[doc]['sentences']):
                tokens = [w for w, _ in sent['words']]
                i += 1
                if self.docvec:
                    yield gensim.models.doc2vec.TaggedDocument(words=tokens, tags=self.offset+i)#["%s_%d" % (doc, sent_i)])
                else:
                    yield tokens


def preproc(text):
    """ Text preprocessing """
    #text = text.lower()#.decode('utf-8')
    text = re.sub(r"([\.:;,\!\?\"\'])", r" \1 ", text)
    return text.split()


def build_tree(dependencies):
    """ Build tree structure from dependency list """
    tree = collections.defaultdict(lambda: [])
    for rel, parent, child in dependencies:
        tree[parent].append(child)
    return tree


def traverse(tree, node='ROOT-0', depth=0):
    """ Traverse dependency tree, calculate token depths """
    tokens = []
    for child in tree[node]:
        tokens.append((child, depth))
        tokens += traverse(tree, child, depth+1)
    return tokens


def save_vectors(filename, inputs, outputs, label_subst):
    """ Export vector features to text file """
    lookup = dict([(y,x) for x,y in label_subst.items()])
    f = open(filename, "w")
    for input, output in zip(inputs, outputs):
        f.write((lookup[output] + ' ' + ' '.join(map(str, input)) + '\n').encode('utf-8'))
    f.close()


def read_file(filename, parses):
    """ Read relation data from JSON """
    relations = []
    all_relations = []
    for row in open(filename):
        rel = json.loads(row)
        doc = parses[rel['DocID']]
        arg1 = get_token_depths(rel['Arg1'], doc)
        arg2 = get_token_depths(rel['Arg2'], doc)
        context = None #get_context(rel, doc, context_size=1)
        # Use for word vector training
        all_relations.append((rel['Sense'], arg1, arg2))
        # Use for prediction (implicit relations only)
        if rel['Type'] in ['Implicit', 'EntRel']:#, 'AltLex']:
            relations.append((rel['Sense'], arg1, arg2, context))
    return (relations, all_relations)


def get_token_depths(arg, doc):
    """ Wrapper for token depth calculation """
    tokens = []
    depths = {}
    for _, _, _, sent_i, token_i in arg['TokenList']:
        if sent_i not in depths:
            depths[sent_i] = dict(traverse(build_tree(doc['sentences'][sent_i]['dependencies'])))
        token, _ = doc['sentences'][sent_i]['words'][token_i]
        try:
            tokens.append((token, depths[sent_i][token+'-'+str(token_i+1)]))
        except KeyError:
            tokens.append((token, None))
    return tokens


def get_context(rel, doc, context_size=2):
    """ Get tokens from context sentences of arguments """
    pretext, posttext = [], []
    for context_i in reversed(range(context_size+1)):
        _, _, _, sent_i, _ = rel['Arg1']['TokenList'][0]
        for token_i, token in enumerate(doc['sentences'][sent_i-context_i]['words']):
            token, _ = token
            if context_i == 0 and token_i >= rel['Arg1']['TokenList'][0][-1]:
                break
            pretext.append(token)
    for context_i in range(context_size+1):
        _, _, _, sent_i, _ = rel['Arg2']['TokenList'][-1]
        try:
            for token_i, token in enumerate(doc['sentences'][sent_i+context_i]['words']):
                token, _ = token
                if context_i == 0 and token_i <= rel['Arg2']['TokenList'][-1][-1]:
                    continue
                posttext.append(token)
        except IndexError:
            pass
    return (pretext, posttext)


def convert_relations(relations, label_subst, m):
    inputs = []
    outputs = []
    rel_dict = collections.defaultdict(lambda: [])
    # Convert relations: word vectors from segment tokens, aggregate to fix-form vector per segment
    for i, rel in enumerate(relations):
        senses, arg1, arg2, context = rel
        if i % 1000 == 0:
            print "Converting relation",i
        for sense in [senses[0]]:
            avg = np.average([d for t, d in arg1 if d is not None])
            # Get tokens and weights
            tokens1 = [(token, 1./(2**depth)) if depth is not None else (token, 0.25) for token, depth in arg1]
            # Get weighted token vectors
            vecs = np.transpose([m[t]*w for t,w in tokens1 if t in m] + [m[t.lower()]*w for t,w in tokens1 if t not in m and t.lower() in m])
            if len(vecs) == 0:
                vecs = m['a']*0
            vec1 = np.array(map(np.average, vecs))
            vec1prod = np.array(map(np.prod, vecs))
            # Get vectors for tokens in context (before arg1)
            """context1 = np.transpose([m[t] for t in context[0] if t in m] + [m[t.lower()] for t in context[0] if t not in m and t.lower() in m])
            if len(context1) == 0:
                context1avg = vec1*0
            else:
                context1avg = np.array(map(np.average, context1))
            """
            #max1 = np.array(map(max, vecs))
            #min1 = np.array(map(min, vecs))
            avg = np.average([d for t, d in arg2 if d is not None])
            # Get tokens and weights
            tokens2 = [(token, 1./(2**depth)) if depth is not None else (token, 0.25) for token, depth in arg2]
            # Get weighted token vectors
            vecs = np.transpose([m[t]*w for t,w in tokens2 if t in m] + [m[t.lower()]*w for t,w in tokens2 if t not in m and t.lower() in m])
            if len(vecs) == 0:
                vecs = m['a']*0
            # Get vectors for tokens in context (after arg2)
            """context2 = np.transpose([m[t] for t in context[1] if t in m] + [m[t.lower()] for t in context[1] if t not in m and t.lower() in m])
            if len(context2) == 0:
                context2avg = vec2*0
            else:
                context2avg = np.array(map(np.average, context2))
            """
            vec2 = np.array(map(np.average, vecs))
            vec2prod = np.array(map(np.prod, vecs))
            #max2 = np.array(map(max, vecs))
            #min2 = np.array(map(min, vecs))
            #final = np.concatenate([vec1,max1,min1,vec2,max2,min2])
            #final = np.concatenate([vec1,vec2])
            ##final = np.concatenate([np.add(vec1prod,vec1), np.add(vec2prod,vec2), context1avg, context2avg])
            final = np.concatenate([np.add(vec1prod,vec1), np.add(vec2prod,vec2)])
            #final = np.concatenate([vec1prod,vec1, vec2prod,vec2])
            if len(final) == 2*len(m['a']):
                inputs.append(final)
            else:
                print "Warning: rel %d has length %d" % (i, len(final))
                if len(vec1) == 0:
                    print "arg1", arg1
                if len(vec2) == 0:
                    print "arg2", arg2
                break
            outputs.append(np.array(label_subst[sense]))
    ## Theanets training from this point on
    inputs = np.array(inputs)
    inputs = inputs.astype(np.float32)
    outputs = np.array(outputs)
    outputs = outputs.astype(np.int32)
    return (inputs, outputs) 


def convert_relations_docvec(relations, label_subst, m):
    inputs = []
    outputs = []
    rel_dict = collections.defaultdict(lambda: [])
    # Convert relations: word vectors from segment tokens, aggregate to fix-form vector per segment
    for i, rel in enumerate(relations):
        senses, arg1, arg2, context = rel
        if i % 1000 == 0:
            print "Converting relation",i
        for sense in senses:
            final = np.concatenate([m.docvecs["SEG_%d.0" % i], m.docvecs["SEG_%d.1" % i]])
            if len(final) > 0:
                inputs.append(final)#fÃƒÂ¼r jede implicit relation Bedeutung (z.B Comparison.Contrast) wird eine Liste in input eingefÃƒÂ¼gt, die vec,max,min fÃƒÂ¼r beide args enthÃƒÂ¤lt
            else:
                continue
            outputs.append(np.array(label_subst[sense]))
    ## Theanets training from this point on
    inputs = np.array(inputs)
    inputs = inputs.astype(np.float32)
    outputs = np.array(outputs)
    outputs = outputs.astype(np.int32)
    return (inputs, outputs)


