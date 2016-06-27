from train_chinese import *
import sys


if __name__ == "__main__":
    input_train, output_train, input_dev, output_dev, label_subst = start_vectors(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    (acc, valid_acc, train_acc,label_subst, predicted_labels) = train_theanet('nag', 0.0001, 0.85, 0.0001, 0.0001, (40, 'lgrelu'), 0.001,5,5, "l2","l2",input_train, output_train, input_dev, output_dev, label_subst)

#e.g. python test_single.py data/zh-01-08-2016-train/parses.json data/zh-01-08-2016-dev/parses.json data/zh-01-08-2016-train/relations.json data/zh-01-08-2016-dev/relations.json data/zh-Gigaword-300.txt
