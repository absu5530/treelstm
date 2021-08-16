# Engineering a Child-Sum Tree LSTM with spaCy Transformer Dependency Trees

This is a modified implementation of the methods proposed in [Improved Semantic Representations From
Tree-Structured Long Short-Term Memory Networks](https://aclanthology.org/P15-1150.pdf) (Tai et al., 2015) to develop 
LSTM network models with dependency trees as inputs. The salient features of this work are an architecture for using 
spaCy dependency trees with BERT transformer embeddings as the input to a tree LSTM, rather than the 
[Stanford Neural Network Dependency Parser](https://www-nlp.stanford.edu/software/nndep.html) (Chen and Manning, 2014) 
with GloVe embeddings as in the original paper.

This implementation is hyperparameter-tuned, trained and tested on the 
[General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) Microsoft Research Paraphrase 
Corpus (MRPC), a sentence paraphrase dataset built from news article that is labeled for whether or not each pair is a 
paraphrase pair.

The primary hypothesis is that BERT embeddings trained and updated in a dependency tree LSTM should give a stronger 
signal in determining whether two sentences are paraphrases of each other as opposed to a plain BERT classifier.

## Training



## What are dependency trees?

Dependency trees are potent representations of sentence structure. They give us important syntactic information that 
cannot easily be encoded in a sequential or tabular format, by delineating for us the relationships between words that 
make up sentences. An example of a dependency tree, for the sentence `The big grey cat is sleeping.`, is 
below:

![](data/dep_tree.png)

In this example the root verb is `sleeping`, which branches out to its dependents, the noun `cat` and the auxiliary 
verb `is`. The noun `cat` further branches out to its dependents, the determiner `The`, and the adjectives `big` and 
`grey`.

Dependency information has a wide variety of applications in Natural Language Processing. Harnessing this information 
can be as simple as using counts of dependency tags or their n-grams as a proxy for syntactic richness. This particular 
implementation is based on research that attempts to harness the inherent tree structure of dependency parsing to the 
advantage of an LSTM network.

## What is BERT?



## What is the model architecture?

## Analysis of Results

## Future Work