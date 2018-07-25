# NER_using_DeepLearning
Named Entity Recognition based on neural networks

I have used standard gmb-2.2.0 annotated dataset.

There are two files where one has decision_tree_classification implementation and the other has keras classifier(neural network) implementation.

I got 95.56% accuracy with decision tree classifier where as 96.34% accuracy with keras classifier.

Features used are word_in_lower_case, pos_tag of the word, last 3 letters of the word, last 2 letters of the word, if the word is in caps, if the word starts with caps ,if the word is in beginning or end of the sentence and same features for the next word or previous word in the sentence according to its place. 


For testing pipeline implementation of vectorizer and keras, Run file 'Neural_network_classifier' with all the requirements installed.It asks for user input to input the test sentence and returns the NER tags for each word in the sentence.

* Create a Virtual environment
```
pip install virtualenv
virtualenv .
```
* Install requiremnts
```
pip install -r requirements.txt

```
* Run classifiers
```
python path(neural_network_classifier.py)
python path(decision_tree_classif.py)
```




Training sentences were only 10000 when I got this accuracy due to lack of computational capabilities.Accuracy would increase when we increase the training data.
