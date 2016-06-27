from train import *
import csv
import datetime
import sys

def test(parses_train_filepath, parses_dev_filepath, relations_train_filepath, relations_dev_filepath, googlevecs_filepath):
    now = datetime.datetime.now()
    dateAsString = now.strftime("%Y-%m-%d %H:%M")
    csvfile = open('Results'+dateAsString+'.csv', 'w')
    fieldnames = ['Test Acc', 'Valid Acc', 'Train Acc', "MinImprov", "Method", "LernR", "Momentum", "Decay", "Regular.", "Hidden"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    input_train, output_train, input_dev, output_dev, label_subst = start_vectors(parses_train_filepath, parses_dev_filepath, 
        relations_train_filepath, relations_dev_filepath, googlevecs_filepath)
    #different parameter options, e.g.:
    method = ['nag']
    min_improvements = [0.001]
    learning_rates = [0.0001]
    w_h = [('l2', 'l1'), ('l1', 'l2'), ('l2','l2'), ("l1", "l1")]
    momentum_alts = [0.1,0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    hidden_alts = [60, 65, 70, 75, 80, 85, 90, 95, 100]
    act_funcs = ['rect:max','prelu','lgrelu']
    d_r = [(0.0001, 0.0001), (0.0001, 0.1)]

    ## more parameter options, e.g.:
    #method = ['nag', 'sgd', 'rprop','rmsprop', 'adadelta', 'hf', 'sample','layerwise']
    #min_improvements = [0.001, 0.005, 0.1, 0.2]
    #w_h = [('l2', 'l1'), ('l1', 'l2'), ('l2','l2'), ("l1", "l1")]
    #learning_rates = [0.0001, 0.0005, 0.001, 0.005]
    #momentum_alts = [0.1,0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    #hidden_alts = [20, 30, 40, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    #act_funcs = ['linear','logistic','tanh+norm:z',  
    #             'softplus', #'softmax', --> is to bad
    #             'relu','rect:min', 'rect:max',
    #             'norm:mean','norm:max', 'norm:std',
    #             'norm:z', 'prelu','lgrelu']

    #decay = [0.0001, 0.1, 0.2, 0.5]
    #regularization = [0.0001, 0.1, 0.2, 0.5]
    #d_r = []
    #for i in decay:
    #    for j in regularization:
    #        d_r.append((i,j))
    
    best_acc = 0
    for h in method:
        for i in min_improvements:
            for j in learning_rates:
                for k in momentum_alts:
                    for l in hidden_alts:
                        for m in act_funcs:
                            for n in w_h:
                                for o in d_r:
                                    (acc, valid_acc, train_acc) = train_theanet(h, j, k, o[0], o[1], (l, m), i, 5,5, n[0], 
                                                                                n[1], input_train, output_train, input_dev, 
                                                                                output_dev, label_subst)
                                    writer.writerow({'Test Acc': round(acc*100,2), 'Valid Acc': round(valid_acc*100,2) , 
                                                     "Train Acc": round(train_acc*100,2),
                                                     "MinImprov": i, "Method": h, "LernR": j,
                                                     "Momentum":k, "Decay":"{0}={1}".format(n[0], o[0]), "Regular.": "{0}={1}".format(n[1], o[1]),
                                                     "Hidden": "({0}, {1})".format(l,m)})
                                    if acc > best_acc:
                                        best_acc = acc
    csvfile.close()


if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
#e.g. python test_parameterSettings.py data/en-01-12-16-train/parses.json data/en-01-12-16-dev/parses.json data/en-01-12-16-train/relations.json data/en-01-12-16-dev/relations.json data/GoogleNews-vectors-negative300.bin


