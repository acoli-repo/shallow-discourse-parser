# -*- coding: utf-8 -*-
import json
import numpy as np
import collections


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


def read_file(filename, parses):
    """ Read relation data from JSON """
    relations = []
    all_relations = []
    for row in open(filename):
        rel = json.loads(row)
        doc = parses[rel['DocID']]
        arg1 = get_token_depths(rel['Arg1'], doc)
        arg2 = get_token_depths(rel['Arg2'], doc)
        context = get_context(rel, doc, context_size=1)
        # Use for word vector training
        all_relations.append((rel['Sense'], arg1, arg2))
        # Use for prediction (implicit relations only)
        if rel['Type'] in ['Implicit', 'EntRel']:#, 'AltLex']:
            relations.append((rel['Sense'], arg1, arg2, context))
    return (relations, all_relations)


def read_file_Org(filename, parses):
    """ Read relation data from JSON """
    relations = []
    all_relations = []
    for row in open(filename):
        rel = json.loads(row)
        doc = parses[rel['DocID']]#get all data in pdtb-parses.json file to the same wsj document
        arg1 = get_token_depths_Org(rel['Arg1'], doc)#of the data for Arg1/2 we only need TokenList in get_token_depth function!
        arg2 = get_token_depths_Org(rel['Arg2'], doc)
        context = None #get_context_Org(rel, doc, context_size=1)
        # Use for word vector training
        all_relations.append((rel['Sense'], arg1, arg2))
        if rel['Type'] not in ['Implicit', 'EntRel']:
            continue
        # Use for prediction (implicit relations only)
        relations.append((rel['Sense'], arg1, arg2, context))
    return (relations, all_relations)


def get_token_depths_Org(arg, doc):
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


def read_file_noSenses(filename, parses):
    """ Read relation data from JSON """
    relations = []
    all_relations = []
    counter = 0
    for row in open(filename):
        rel = json.loads(row)
        doc = parses[rel['DocID']]
        arg1 = get_token_depths_Org(rel['Arg1'], doc)
        arg2 = get_token_depths_Org(rel['Arg2'], doc)
        context = None #get_context_Org(rel, doc, context_size=1)
        # Use for word vector training
        all_relations.append((rel['Sense'], arg1, arg2))
        if rel["Connective"]["TokenList"] == []:
            counter+=1
            relations.append((rel['Sense'], arg1, arg2, context))
        else:
            continue
    return (relations, all_relations)


def get_token_depths(arg, doc):
    """ Wrapper for token depth calculation """
    tokens = []
    depths = {}
    for sent_i, token_i in arg['TokenList']:
        if sent_i not in depths:
            depths[sent_i] = dict(traverse(build_tree(doc['sentences'][sent_i]['dependencies'])))
        token, _ = doc['sentences'][sent_i]['words'][token_i]
        try:
            tokens.append((token, depths[sent_i][token+'-'+str(token_i+1)]))
        except KeyError:
            tokens.append((token, None))
    return tokens

def get_context_Org(rel, doc, context_size=2):
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


def get_context(rel, doc, context_size=2):
    """ Get tokens from context sentences of arguments """
    pretext, posttext = [], []
    for context_i in reversed(range(context_size+1)):
        if rel['Arg1']['TokenList'] == []:
            pretext.append("")
        else:
            sent_i, _ = rel['Arg1']['TokenList'][0]
            for token_i, token in enumerate(doc['sentences'][sent_i-context_i]['words']):
                token, _ = token
                if context_i == 0 and token_i >= rel['Arg1']['TokenList'][0][-1]:
                    break
                pretext.append(token)
    for context_i in range(context_size+1):
        if rel['Arg2']['TokenList'] == []:
            posttext.append("")
        else:
            sent_i, _ = rel['Arg2']['TokenList'][-1]
            try:
                for token_i, token in enumerate(doc['sentences'][sent_i+context_i]['words']):
                    token, _ = token
                    if context_i == 0 and token_i <= rel['Arg2']['TokenList'][-1][-1]:
                        continue
                    posttext.append(token)
            except IndexError:
                pass
    #print(pretext, posttext)
    return (pretext, posttext)



def convert_relations_noSenses(relations, label_subst, m):
    inputs = []
    outputs = []
    rel_dict = collections.defaultdict(lambda: [])
    # Convert relations: word vectors from segment tokens, aggregate to fix-form vector per segment
    for i, rel in enumerate(relations):
        senses, arg1, arg2, context = rel
        if i % 1000 == 0:
            print "Converting relation",i
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
        """context1 = np.transpose([m[t] for t in context[0] if t in m] + [m[t.lower()] for t in context[0] if t not in m a$
        if len(context1) == 0:
            context1avg = vec1*0
        else:
            context1avg = np.array(map(np.average, context1))
        """
        avg = np.average([d for t, d in arg2 if d is not None])
        # Get tokens and weights
        tokens2 = [(token, 1./(2**depth)) if depth is not None else (token, 0.25) for token, depth in arg2]
        # Get weighted token vectors
        vecs = np.transpose([m[t]*w for t,w in tokens2 if t in m] + [m[t.lower()]*w for t,w in tokens2 if t not in m and t.lower() in m])
        if len(vecs) == 0:
            vecs = m['a']*0
        """context1 = np.transpose([m[t] for t in context[0] if t in m] + [m[t.lower()] for t in context[0] if t not in m a$
        if len(context1) == 0:
            context1avg = vec1*0
        else:
            context1avg = np.array(map(np.average, context1))
        """
        avg = np.average([d for t, d in arg2 if d is not None])
        # Get tokens and weights
        tokens2 = [(token, 1./(2**depth)) if depth is not None else (token, 0.25) for token, depth in arg2]
        # Get weighted token vectors
        vecs = np.transpose([m[t]*w for t,w in tokens2 if t in m] + [m[t.lower()]*w for t,w in tokens2 if t not in m and t.lower() in m])
        if len(vecs) == 0:
            vecs = m['a']*0
        # Get vectors for tokens in context (after arg2)
        """context2 = np.transpose([m[t] for t in context[1] if t in m] + [m[t.lower()] for t in context[1] if t not in m a$
        if len(context2) == 0:
            context2avg = vec2*0
        else:
           context2avg = np.array(map(np.average, context2))
        """
        vec2 = np.array(map(np.average, vecs))
        vec2prod = np.array(map(np.prod, vecs))
        final = np.concatenate([np.add(vec1prod,vec1), np.add(vec2prod,vec2)])
        if len(final) == 2*len(m['a']):
            inputs.append(final)
        else:
            print "Warning: rel %d has length %d" % (i, len(final))
            if len(vec1) == 0:
                print "arg1", arg1
            if len(vec2) == 0:
                print "arg2", arg2
            break
    inputs = np.array(inputs)
    inputs = inputs.astype(np.float32)
    outputs = np.array(outputs)
    outputs = outputs.astype(np.int32)
    return (inputs, outputs)


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
            final = np.concatenate([np.add(vec1prod,vec1), np.add(vec2prod,vec2)])
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
                inputs.append(final)
            else:
                continue
            outputs.append(np.array(label_subst[sense]))
    inputs = np.array(inputs)
    inputs = inputs.astype(np.float32)
    outputs = np.array(outputs)
    outputs = outputs.astype(np.int32)
    return (inputs, outputs)
