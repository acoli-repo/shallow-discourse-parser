import json
import ast
import os

def parses_load(parses_file):
    ''' open parses file '''
    with open(parses_file) as data_file:    
        data = json.load(data_file)
    return data


def get_words_dict(parses_file):
    ''' create a dict with DocID as keys and list of triples (sent_id, token_id, token) for each word as values'''
    data = parses_load(parses_file)
    wsj_id_list = [i for i in data.keys()]
    docID_words_dict = dict()
    for wsj_id in wsj_id_list:
        tokenlist = []
        words = []
        counter_sentence = 0
        counter_token=0
        for i in data[wsj_id]["sentences"]:
            for j in i["words"]:
                words.append((counter_sentence, counter_token, j[0])) #append triple (sent_id, token_id, token)
                counter_token+=1
            counter_sentence+=1
            counter_token=0#counting token_id in sentence, not in whole text
        docID_words_dict[wsj_id] = words
    return docID_words_dict


def subst_id_words(filename, parses_file):
    ''' substitute the TokenList of [filename] by the new TokenList (with list [sent_id, token_id] for each token) '''
    new_rows = []
    counter = 0
    parses_dict = get_words_dict(parses_file)
    for row in open(filename, "r"):
        counter+=1
        new_row = ""
        data = json.loads(row)
        wsj_id = data["DocID"]
        words_ids_arg1 = data["Arg1"]["TokenList"]
        words_ids_arg2 = data["Arg2"]["TokenList"]
        tokenlist_arg1 = [[parses_dict[wsj_id][i][0], parses_dict[wsj_id][i][1]] for i in words_ids_arg1]
        tokenlist_arg2 = [[parses_dict[wsj_id][i][0], parses_dict[wsj_id][i][1]] for i in words_ids_arg2]
        data['Arg1']['TokenList'] = tokenlist_arg1
        data['Arg2']['TokenList'] = tokenlist_arg2
        new_row=json.dumps(data)
        new_rows.append(new_row)
    new_filename = filename.split(".json")[0]+"_modifiedTokenList.json"
    g = open(new_filename, "w")
    for i in new_rows:
        g.writelines(i+"\n")
    g.close()
    return new_filename
