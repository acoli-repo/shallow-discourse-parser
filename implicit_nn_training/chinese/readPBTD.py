import json
import ast
import os



def parses_load(parses_file):
    with open(parses_file) as data_file:    
        data = json.load(data_file)
        #data.update(json.load(open('parses_dev.json')))
    return data


def get_words_dict(parses_file):
    data = parses_load(parses_file)
    wsj_id_list = [i for i in data.keys()]
    docID_words_dict = dict()
    #words = []
    #tokenlist = []
    #counter_sentence = 0
    #counter_token = 0
    for wsj_id in wsj_id_list:
        tokenlist = []
        words = []
        counter_sentence = 0
        counter_token=0
        for i in data[wsj_id]["sentences"]:
            for j in i["words"]:
                words.append((counter_sentence, counter_token, j[0])) #append Tupel (sent_id, token_id, token)
                counter_token+=1
            counter_sentence+=1
            counter_token=0#counting token_id in sentence, not in whole text
        docID_words_dict[wsj_id] = words
    return docID_words_dict


def get_words(wsj_id, parses_file):
    data = parses_load(parses_file)
    words = []
    tokenlist = []
    counter_sentence = 0
    counter_token = 0
    for i in data[wsj_id]["sentences"]:
        for j in i["words"]:
            words.append((counter_sentence, counter_token, j[0])) #append Tupel (sent_id, token_id, token)
            counter_token+=1
        counter_sentence+=1
        counter_token=0#counting token_id in sentence, not in whole text
    return words

def subst_id_words(filename, parses_file):
    new_rows = []
    counter = 0
    parses_dict = get_words_dict(parses_file)
    for row in open(filename, "r"):
        counter+=1
        new_row = ""
        data = json.loads(row)
        #if data["Type"] != "Implicit":
         #   new_row=json.dumps(data)
	  #  new_rows.append(new_row)
        wsj_id = data["DocID"]
        words_ids_arg1 = data["Arg1"]["TokenList"]
        words_ids_arg2 = data["Arg2"]["TokenList"]
        #words = get_words(wsj_id)#hier ein einziges dict machen
        #tokenlist_arg1 = [[words[i][0], words[i][1]] for i in words_ids_arg1]
        #tokenlist_arg2 = [[words[i][0], words[i][1]] for i in words_ids_arg2]
        tokenlist_arg1 = [[parses_dict[wsj_id][i][0], parses_dict[wsj_id][i][1]] for i in words_ids_arg1]
        tokenlist_arg2 = [[parses_dict[wsj_id][i][0], parses_dict[wsj_id][i][1]] for i in words_ids_arg2]
        #arg1_words = [words[i][2] for i in words_ids_arg1]
        #arg2_words = [words[i][2] for i in words_ids_arg2]      
        #data['Arg1']['RawText'] = arg1_words
        #data['Arg2']['RawText'] = arg2_words
        #data['Arg1']['TokenList'] = tokenlist_arg1
        #data['Arg2']['TokenList'] = tokenlist_arg2
        if data["Arg1"]['TokenList'] == []:
            data["Arg1"]['TokenList'] == [""]
            print("found arg1")
        else:
            data['Arg1']['TokenList'] = tokenlist_arg1

        if data["Arg2"]['TokenList'] == []:
            print("found arg2")
            data["Arg2"]['TokenList'] == [""]
        else:
            data['Arg2']['TokenList'] = tokenlist_arg2
        new_row=json.dumps(data)
        new_rows.append(new_row)
    new_filename = filename.split(".json")[0]+"_modifiedTokenList.json"
    g = open(new_filename, "w")
    for i in new_rows:
        g.writelines(i+"\n")
    g.close()
    return new_filename



