from train import *
import sys


if __name__ == "__main__":
    ''' train the neural network with a given parameter setting'''
    input_train, output_train, input_dev, output_dev, label_subst = start_vectors(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    ## to know where to get the parses/relations file, please read README "Data Requirements"
    # sys.argv[1] := absolute path to train parses json file 
    # sys.argv[2] := absolute path to dev parses json file
    # sys.argv[3] := absolute path to train relations file
    # sys.argv[4] := absolute path to dev relations file
    # sys.argv[5] := absolute path to GoogleNews-vectors-negative300.bin file(binary file)
    (acc, valid_acc, train_acc) = train_theanet('nag', 0.0001, 0.6, 0.0001, 0.0001, (60, 'lgrelu'), 0.001,5,5, "l1", "l2", input_train, output_train, input_dev, output_dev, label_subst)
    #the argumetns of train_theanets are: method, learning_rate, momentum, decay, regularization, hidden, min_improvement, validate_every, patience, weight_lx, hidden_lx, input_train, output_train, input_dev, output_dev, label_subst


#e.g. python test_single.py data/en-01-12-16-train/parses.json data/en-01-12-16-dev/parses.json data/en-01-12-16-train/relations.json data/en-01-12-16-dev/relations.json data/GoogleNews-vectors-negative300.bin
