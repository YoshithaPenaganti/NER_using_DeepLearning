# NER_using_DeepLearning
Named Entity Recognition based on neural networks

I have used standard gmb-2.2.0 annotated dataset.

There are two files where one has decision_tree_classification implementation and the other has keras classifier(neural network) implementation.

I got 88% accuracy with decision tree classifier where as 95% accuracy with keras classifier.


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
python path(Neural_network_classifier)
python path(decision_tree_classif.py)
```




Training sentences were only 10000 when I got this accuracy due to lack of computational capabilities.Accuracy would increase when we increase the training data.
