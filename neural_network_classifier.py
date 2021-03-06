import os
import collections

from nltk import pos_tag,word_tokenize

ner_tags = collections.Counter()

corpus_root = "gmb-2.2.0"
def iob(list_annotated):

    list_iob_tokens = []
    for idx, token in enumerate(list_annotated):
        tag, word, entity = token

        if entity != 'O':
            if idx == 0:
                entity = "B-" + entity
            elif list_annotated[idx - 1][2] == entity:
                entity = "I-" + entity
            else:
                entity = "B-" + entity
        list_iob_tokens.append((tag, word, entity))
    return list_iob_tokens
def read_data_from_folder(corpus_root):
    processed_sentences = []
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]

                        list_tokens = []

                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]

                            if ner != 'O':
                                ner = ner.split('-')[0]

                            list_tokens.append((word, tag, ner))

                        processed_sentences.append(iob(list_tokens))
    return processed_sentences


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]

data =read_data_from_folder(corpus_root)
sent2features(data[0])


sentences_for_training = data[:10000]
sentences_for_testing = data[10000:11000]

def sentence_split(list_of_sentences):
    data_split = []
    for sentence in list_of_sentences:
        for word in sentence:
            data_split.append(word)
    return data_split

X_train = sentence_split([sent2features(sentence) for sentence in sentences_for_training])

y_train = sentence_split([sent2labels(sentence) for sentence in sentences_for_training])

X_test = sentence_split([sent2features(sentence) for sentence in sentences_for_testing])

y_test = sentence_split([sent2labels(sentence) for sentence in sentences_for_testing])


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras.wrappers.scikit_learn import KerasClassifier

# load your data

# create a function that returns a model, taking as parameters things you
# want to verify using cross-valdiation and model selection
def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(17, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# wrap the model using the function you created
keras_classif = KerasClassifier(build_fn=create_model,verbose=0)



param_grid = {
    'clf__optimizer':['rmsprop','adam','adagrad'],
    'clf__epochs':[4,8],
    'clf__dropout':[0.1,0.2],
    'clf__kernel_initializer':['glorot_uniform','normal','uniform']
}

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', keras_classif)])

clf.fit(X_train,y_train)

print('Training completed')


print("Accuracy:", clf.score(X_test, y_test))

test_labelled_sentence =sent2features(pos_tag(word_tokenize(input('Give the test sentence:'))))
for word in test_labelled_sentence:
    print('{} - {}'.format(word['word.lower()'],clf.predict(word)))
