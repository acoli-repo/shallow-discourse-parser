from train_chinese_pickle import *
import csv
import datetime
import sys


# example for parameter (learning_rate, min_improvement, method are fix in this code)
parameter = [(0.1, 95, "prelu", "l2", 0.0001, "l1", 0.1), (0.3, 100, "prelu", "l2", 0.0001, "l2", 0.1 ),
(0.35, 95, "rect:max", "l1", 0.0001, "l1", 0.1), (0.35, 95, "prelu", "l2", 0.0001, "l1", 0.1), 
(0.35, 100, "prelu", "l2", 0.0001, "l1", 0.1), (0.4, 80, "prelu", "l2", 0.0001, "l1", 0.1)]

def test(parses_train_filepath, parses_dev_filepath, relations_train_filepath, relations_dev_filepath, gigaword_filepath, parameter):
    now = datetime.datetime.now()
    dateAsString = now.strftime("%Y-%m-%d %H:%M")
    csvfile = open('Results_pickle'+dateAsString+'.csv', 'w')
    fieldnames = ['VectorTraining','NN Training', 'Test Acc', 'Valid Acc', 'Train Acc', "MinImprov", "Method", "LernR", "Momentum", "Decay", "Regular.", "Hidden"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    counter_vec = 0
    counter_nn = 0
    for iter1 in range(1,4):
        #train vectors 3x
        counter_vec+=1
        writer.writerow({})
        input_train, output_train, input_dev, output_dev, label_subst = start_vectors(counter_vec, parses_train_filepath, 
        parses_dev_filepath, relations_train_filepath, relations_dev_filepath, gigaword_filepath)
        for iter2 in range(len(parameter)*5):
            #for each trained vectors train each NN parameter combination 5x 
            counter_nn+=1
            if iter2%5 == 0:
                triple = parameter[iter2//5]
            (acc, valid_acc, train_acc) = train_theanet(counter_vec, counter_nn, 'nag', 0.0001, triple[0],
                                                                             triple[4], triple[6],(triple[1],triple[2]), 0.001, 5,5, 
                                                                             triple[3], triple[5], input_train, output_train, input_dev,
                                                                             output_dev, label_subst)
            writer.writerow({'VectorTraining': counter_vec ,'NN Training': counter_nn,  'Test Acc': round(acc*100,2), 'Valid Acc': round(valid_acc*100,2) , 
                   "Train Acc": round(train_acc*100,2), "MinImprov": 0.001, "Method": "nag", "LernR": 0.0001,"Momentum":triple[0], 
                   "Decay":"{0}={1}".format(triple[3], triple[4]), "Regular.": "{0}={1}".format(triple[5],triple[6]), "Hidden": 
                   "({0}, {1})".format(triple[1],triple[2])})
    csvfile.close()


if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], parameter)


#e.g. python test_parameterSettings_pickle.py data/zh-01-08-2016-train/parses.json data/zh-01-08-2016-dev/parses.json data/zh-01-08-2016-train/relations.json data/zh-01-08-2016-dev/relations.json data/zh-Gigaword-300.txt
