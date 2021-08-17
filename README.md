# Engineering a Child-Sum Tree-LSTM with spaCy Transformer Dependency Trees

This is a modified implementation of the methods proposed in [Improved Semantic Representations From
Tree-Structured Long Short-Term Memory Networks](https://aclanthology.org/P15-1150.pdf) (Tai et al., 2015) to develop 
LSTM network models with dependency trees as inputs, or Dependency Tree-LSTMs. The salient features of this work are an 
architecture for using spaCy dependency trees with BERT transformer embeddings as the input to a tree LSTM, rather than 
the [Stanford Neural Network Dependency Parser](https://www-nlp.stanford.edu/software/nndep.html) (Chen and Manning, 
2014) with GloVe embeddings as in the original paper.

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

![Composition of memory cell c and hidden staten h of a Tree-LSTM cell unit](data/treelstmcell.png)

The foundation of the Dependency-Tree LSTM is the Child-Sum Tree-LSTM cell unit. The difference between the Tree-LSTM 
cell unit and a regular standard LSTM unit is that in a Tree-LSTM unit, the gating vectors `i` and `f` and the memory 
cell `c` are dependent on the hidden and cell states of multiple child units. There are multiple forget gates `f`, one 
for each child unit. This allows the Tree-LSTM unit to selectively exclude information from each child unit. In this 
way, it could learn to prioritize certain dependency children more than others, e.g. adverbial clauses vs. determiners 
in trying to determine semantic relatedness.

The input `x` for each Tree-LSTM cell unit, in our case, is the spaCy BERT embedding for the particular token from the 
pooled output of the spaCy BERT model.

![](data/treelstm.png)

There are three variations in model architecture that are implemented in this work for comparison, the `Similarity Tree 
LSTM`, which is composed solely of Child-Sum Tree-LSTM cells, the `BERT Classifier` model, which is a plain BERT 
classifier model consisting of a dropout and dense layer applied on the pooled output of a BERT language model, and a 
combined `BERT + Similarity Tree LSTM` model. The architecture of these models is illustrated in the diagram above.

## Analysis of Results

## Future Work