The Frankfurt Shallow Discourse Parser
====================================

Developed at the Applied Computational Linguistics Lab (ACoLi), Goethe University Frankfurt am Main, Germany.

This repository hosts the shallow discourse parser described in: [Do We Really Need All Those Rich Linguistic Features? A Neural Network-Based Approach to Implicit Sense Labeling](http://www.conll.org/cfp-2016)

Niko Schenk, Christian Chiarcos, Samuel Rönnqvist, Kathrin Donandt, Evgeny A. Stepanov and Giuseppe Riccardi. "Do We Really Need All Those Rich Linguistic Features? A Neural Network-Based Approach to Implicit Sense Labeling". In *Proceedings of the Twentieth Conference on Computational Natural Language Learning - Shared Task, CoNLL 2016*. 2016.

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


## Software Requirements

The parser runs on a Linux environment using Python 2.7.

Please install the following prerequisites in the given order:
$ pip install Cython 
$ pip install --upgrade gensim
$ pip install theanets
$ pip install nose-parameterized
$ pip install -U scikit-learn


If you encounter issues with the numpy version, you can install all the software on a virtual environment (http://docs.python-guide.org/en/latest/dev/virtualenvs/):

$ pip install virtualenv
$ cd shallow-discourse-parser
$ virtualenv venv
$ source venv/bin/activate
$ pip install Cython
$ pip install --upgrade gensim
$ pip install theanets
$ pip install nose-parameterized
$ pip install -U scikit-learn







## Data Requirements

Please copy the following data into the data/ directory:

- Penn Discourse TreeBank (PDTB) 2.0, a 1-million-word Wall Street Journal corpus; there is a train directory and a dev directory (named en-01-12-16-train/ and en-01-12-16-dev/, respectively). They have to include the parses and relations files (normally called pdtb-parses.json/pdtb-relations.json or parses.json/relations.json) (http://www.cs.brandeis.edu/~clp/conll16st/rules.html). Simply, place these two folders into the data/ directory.
- GoogleNews-vectors-negative300 (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?pref=2&pli=1)


