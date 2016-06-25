Shallow Discourse Parser for English
====================================

This repository hosts the shallow discourse parser described in: [Do We Really Need All Those Rich Linguistic Features? A Neural Network-Based Approach to Implicit Sense Labeling](http://www.conll.org/cfp-2016)

@inproceedings{schenk-EtAl:2016:CoNLL-STSDP,
  author    = {Niko Schenk, Christian Chiarcos, Kathrin Donandt, Samuel Rönnqvist, Evgeny A. Stepanov,  Giuseppe Riccardi},
  title     = {{Do We Really Need All Those Rich Linguistic Features? A Neural Network-Based Approach to Implicit Sense Labeling}},
  booktitle = {Proceedings of the Twentieth Conference on Computational Natural Language Learning - Shared Task, CoNLL 2016},
  month     = {August},
  year      = {2016},
  address   = {Berlin, Germany},
  publisher = {Association for Computational Linguistics}
}



## Software Requirements

The parser runs on a Linux Environment.

Please install the following prerequisites in the given order:
- python 2.7
- cython (http://docs.cython.org/src/quickstart/install.html)
- gensim (https://radimrehurek.com/gensim/install.html)
- theanets (https://pypi.python.org/pypi/theanets)


## Data Requirements

Please put the following data into the data/ directory:

- Penn Discourse TreeBank (PDTB) 2.0, a 1-million-word Wall Street Journal corpus; there is a train directory and a dev directory (named en-01-12-16-train/ and en-01-12-16-dev/). They have to have the parses and relations file (normally named pdtb-parses.json/pdtb-relations.json or parses.json/relations.json) (http://www.cs.brandeis.edu/~clp/conll16st/rules.html). Put these two directories into the data/ directory 
- GoogleNews-vectors-negative300 (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?pref=2&pli=1)



## Instructions to run the code

#### (1) Train a neural model with a given parameter setting

$ python test_single.py [path to the train parses file] [path to the dev parses file] [path to the train relations file] [path to the dev relations file] [path to the GoogleNews-vectors-negative300.bin file (binary)]

e.g. $ python test_single.py data/en-01-12-16-train/parses.json data/en-01-12-16-dev/parses.json data/en-01-12-16-train/relations.json data/en-01-12-16-dev/relations.json data/GoogleNews-vectors-negative300.bin

#### (2) Train the neural network with different parameters to find an optimized setting

(takes the same 5 input arguments as test_single.py does)

$ python test_parameterSettings.py [path to the train parses file (e.g. data/en-01-12-16-train/parses.json)] [path to the dev parses file] [path to the train relations file (e.g. data/en-01-12-16-dev/relations.json)] [path to the dev relations file] [path to the GoogleNews-vectors-negative300.bin file (binary)]

e.g. $ python test_parameterSettings.py data/en-01-12-16-train/parses.json data/en-01-12-16-dev/parses.json data/en-01-12-16-train/relations.json data/en-01-12-16-dev/relations.json data/GoogleNews-vectors-negative300.bin

! You can increase the number of values for the parameters in the code, see the lines 25-43
- The results for each parameter setting is saved to Results[date].csv

#### (3) Save the models for different paramter settings

$ python test_parameterSettings_pickle.py [path to the train parses file (e.g. data/en-01-12-16-train/parses.json)] [path to the dev parses file] [path to the train relations file (e.g. data/en-01-12-16-dev/relations.json)] [path to the dev relations file] [path to the GoogleNews-vectors-negative300.bin file (binary)]

e.g. $ python test_parameterSettings_pickle.py data/en-01-12-16-train/parses.json data/en-01-12-16-dev/parses.json data/en-01-12-16-train/relations.json data/en-01-12-16-dev/relations.json data/GoogleNews-vectors-negative300.bin

- Running this script will save the model files to a pickles/ directory
- You can choose the parameter setting for which you want to save the models, changing the parameter list at line 7.
- There are three files for each trained neural network: m_x.py , neuralnetwork_x_y_.save and label_subst_x_y_.py.
- m_x.py is the model vor the word embeddings which the neural network uses; neuralnetwork_x_y_.save is the neural network and label_subst_x_y_.py is a substitution dictionary for relation labels to integers
- For each word_embedding model, we train a the neural network with a certain parameter setting five times, that is why there are 5 neuralnetwork_x_y_.save and label_subst_x_y_.py for each m_x.py
- To be able to identify which model gave which accuracy results, you can look into the file 'Results_pickle[date].csv'. The first two colums indicate the id of the word embedding model and the id of the which neural network models which belong to this word embedding model


#### (3) Load a trained model and run the parser to classify implicit relations

$ python re-classify-implicit_entrel_senses_nn.py [path to file of which implicit senses should be reclassified] [path to output of reclassification] [path to parses file]

e.g. $ python re-classify-implicit_entrel_senses_nn.py output_combined.json output_final.json data/en-01-12-16-dev/parses.json

- Once you have trained the model you wish to use for classifying the implicit relations, you have to move the three model files to the current working directory.
- You have to rename your models "m_best.pickle", "neuralnetwork_best.pickle" and "label_subst_best.pickle".
- We included our model files here, so you can first test the relcassification with our models. 
- An example of a file whose implicit senses should be reclassified is "output_combined.json". It is the combined output of the best Conll15 Shared Task Systems for the development set relations of the Conll16 Shared Task data. So run $ python re-classify-implicit_entrel_senses_nn.py output_combined.json my_output.json [your absolute path to the parses file of the dev set]

