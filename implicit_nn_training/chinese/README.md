Shallow Discourse Parser for Chinese
====================================

This repository hosts the shallow discourse parser described in: [Do We Really Need All Those Rich Linguistic Features? A Neural Network-Based Approach to Implicit Sense Labeling](http://www.conll.org/cfp-2016)

```
@inproceedings{schenk-EtAl:2016:CoNLL-STSDP,
  author    = {Niko Schenk, Christian Chiarcos, Samuel Rönnqvist, Kathrin Donandt, Evgeny A. Stepanov,  Giuseppe Riccardi},
  title     = {{Do We Really Need All Those Rich Linguistic Features? A Neural Network-Based Approach to Implicit Sense Labeling}},
  booktitle = {Proceedings of the Twentieth Conference on Computational Natural Language Learning - Shared Task, CoNLL 2016},
  month     = {August},
  year      = {2016},
  address   = {Berlin, Germany},
  publisher = {Association for Computational Linguistics}
}
```



## Data Requirements

Please put the following data into the data/ directory:

- Chinese Discourse Treebank (CDTB) (http://www.cs.brandeis.edu/~clp/conll16st/rules.html); there is a train directory and a dev directory (named zh-01-08-2016-train/ and zh-01-08-2016-dev/). They have to have the parses and relations file (normally named parses.json/relations.json). Put these two directories into the data/ directory 
- zh-Gigaword-300.txt (http://www.cs.brandeis.edu/~clp/conll16st/data/zh-Gigaword-300.txt)



## Instructions to run the code

#### (1) Train a neural model with a given parameter setting
```
$ python test_single.py [path to the train parses file] [path to the dev parses file] [path to the train relations file] [absolute path to the dev relations file] [path to the  zh-Gigaword-300.txt file (.txt)]
```
e.g. $ python test_single.py data/zh-01-08-2016-train/parses.json data/zh-01-08-2016-dev/parses.json data/zh-01-08-2016-train/relations.json data/zh-01-08-2016-dev/relations.json data/zh-Gigaword-300.txt

#### (2) Train the neural network with different parameters to find an optimized setting

(takes the same 5 input arguments as test_single.py does)
```
$ python test_parameterSettings.py [path to the train parses file (e.g. data/zh-01-08-2016-train/parses.json)] [path to the dev parses file] [path to the train relations file (e.g. data/zh-01-08-2016-train/relations.json)] [path to the dev relations file] [path to the zh-Gigaword-300.txt file (.txt)]
```
e.g. python test_parameterSettings.py data/zh-01-08-2016-train/parses.json data/zh-01-08-2016-dev/parses.json data/zh-01-08-2016-train/relations.json data/zh-01-08-2016-dev/relations.json data/zh-Gigaword-300.txt

! You can increase the number of values for the parameters in the code
- The results for each parameter setting is saved to Results[date].csv

#### (3) Save the models for different paramter settings
```
$ python test_parameterSettings_pickle.py [path to the train parses file (e.g. data/en-01-12-16-train/parses.json)] [path to the dev parses file] [path to the train relations file (e.g. data/zh-01-08-2016-train/relations.json)] [path to the dev relations file] [path to the zh-Gigaword-300.txt file]
```
e.g. $ python test_parameterSettings_pickle.py data/zh-01-08-2016-train/parses.json data/zh-01-08-2016-dev/parses.json data/zh-01-08-2016-train/relations.json data/zh-01-08-2016-dev/relations.json data/zh-Gigaword-300.txt

- Running this script will save the model files to a pickles/ directory
- You can choose the parameter setting for which you want to save the models, changing the parameter list at line 8.
- There are three files for each trained neural network: m_x.py , neuralnetwork_x_y_.save and label_subst_x_y_.py.
- m_x.py is the model vor the word embeddings which the neural network uses; neuralnetwork_x_y_.save is the neural network and label_subst_x_y_.py is a substitution dictionary for relation labels to integers
- For each word_embedding model, we train a the neural network with a certain parameter setting five times, that is why there are 5 neuralnetwork_x_y_.save and label_subst_x_y_.py for each m_x.py
- To be able to identify which model gave which accuracy results, you can look into the file 'Results_pickle[date].csv'. The first two colums indicate the id of the word embedding model and the id of the neural network models belonging to this word embedding model


#### (4) Load a trained model and run the parser to classify implicit relations
```
$ python re-classify-implicit_entrel_senses_nn_chineses.py [path to file of which implicit senses should be reclassified] [path to output of reclassification] [path to parses file]
```
e.g. $ python re-classify-implicit_entrel_senses_nn_chineses.py input_to_be_reclassified_for_implicit_senses.json output_final.json data/zh-01-08-2016-dev/parses.json

- Once you have trained the model you wish to use for classifying the implicit relations, you have to move the three model files to the current working directory.
- You have to rename your models "m_best.pickle", "neuralnetwork_best.pickle" and "label_subst_best.pickle".
- We included our model files here, so you can first test the relcassification with our models. 
