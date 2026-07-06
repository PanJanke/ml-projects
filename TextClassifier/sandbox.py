import string

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from Classifier import Classifier

list = []
labels = []
with open("dataset/allan_poe.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip().lower()
        if line != "":
            line = line.translate(str.maketrans('', '', string.punctuation))
            list.append(line)
            labels.append(0)
with open("dataset/robert_frost.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip().lower()
        if line != "":
            line = line.translate(str.maketrans('', '', string.punctuation))
            list.append(line)
            labels.append(1)

train_text, test_text, Ytrain, Ytest = train_test_split(list, labels) # dzięki temu że teraz dzielimy dataset,a nie po mapowaniu słów na liczby,
                                                                                     # w test_text mogę znaleźć się słowa które nie pojawiły się w train_text
print(len(Ytrain),len(Ytest))
print(train_text[:5])
print(Ytrain[:5])

idx = 1
word2idx = {'<unk>':0} # słowo którego nie ma w słowniku
for text in train_text:
    for word in text.split(" "):
        if word not in word2idx:
            word2idx[word] = idx
            idx += 1
print(len(word2idx))

train_text_int = []
test_text_int = []

for text in train_text:
    tokens = text.split(" ")
    line_as_int = [word2idx[token] for token in tokens]
    train_text_int.append(line_as_int)

for text in test_text:
    tokens = text.split(" ")
    line_as_int = [word2idx.get(token,0) for token in tokens] # jeśli słowa nie ma w słowniku to zamieniamy je na 0 : <unk>
    test_text_int.append(line_as_int)

print(train_text_int[100:105])

V = len(word2idx)
A0 = np.ones((V,V))
pi0 = np.ones(V)

A1 = np.ones((V,V))
pi1 = np.ones(V)  # .ones() tworzy macierz wypełnioną jedynkami, odrazu dodajemy -  add one smoothing.

def compute_counts(text_as_int, A, pi):
    for tokens in text_as_int:
        last_idx = None
        for idx in tokens:
            if last_idx is None:
                pi[idx] += 1
            else:
                A[last_idx, idx] += 1

            last_idx = idx

compute_counts([t for t,y in zip(train_text_int,Ytrain) if y==0], A0, pi0)
compute_counts([t for t,y in zip(train_text_int,Ytrain) if y==1], A1, pi1)

A0 / A0.sum(axis=1, keepdims=True)
pi0 / pi0.sum()

logA0 = np.log(A0)
logpi0 = np.log(pi0)
logA1 = np.log(A1)
logpi1 = np.log(pi1)

#sprawdzamy rozkład klas
count0 = sum(1 for y in Ytrain if y==0)
count1 = sum(1 for y in Ytrain if y==1)
total = len(Ytrain)
p0 = count0 / total
p1 = count1 / total

logp0 = np.log(p0)
logp1 = np.log(p1)

print(f"Class 0: {p0}, Class 1: {p1}")

clf = Classifier([logA0, logA1],[logpi0, logpi1],[logp0, logp1])

Ptrain = clf.predict(train_text_int)
print(f"train accuracy: {np.mean(Ptrain==Ytrain)}")
Ptest = clf.predict(test_text_int)
print(f"test accuracy: {np.mean(Ptest==Ytest)}")

cm = confusion_matrix(Ytrain,Ptrain)
print(cm)
cm_test = confusion_matrix(Ytest,Ptest)
print(cm_test)
f1_train = f1_score(Ytrain, Ptrain)
print(f"F1 score train: {f1_train}")
f1_test = f1_score(Ytest, Ptest)
print(f"F1 score test: {f1_test}")