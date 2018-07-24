import os
import collections

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


sentences_for_training = data[:1000]
sentences_for_testing = data[1000:1100]

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




from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))])

clf.fit(X_train,y_train)  # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)

print('Training completed')


print("Accuracy:", clf.score(X_test, y_test))