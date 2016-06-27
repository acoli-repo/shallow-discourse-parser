ACoLi Shallow Discourse Parser
====================================

Applied Computational Linguistics Lab (ACoLi)
Goethe University Frankfurt

This repository hosts the shallow discourse parser described in: [Do We Really Need All Those Rich Linguistic Features? A Neural Network-Based Approach to Implicit Sense Labeling](http://www.conll.org/cfp-2016)

@inproceedings{schenk-EtAl:2016:CoNLL-STSDP,
  author    = {Niko Schenk, Christian Chiarcos, Samuel Rönnqvist, Kathrin Donandt, Evgeny A. Stepanov,  Giuseppe Riccardi},
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


