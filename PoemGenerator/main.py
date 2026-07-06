import numpy as np
import string
np.random.seed(1234)

initial = {}
first_order = {}
second_order = {}
V = set()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def add2dict(d, key, value):
    if key not in d:
        d[key] = []
    d[key].append(value)

for line in open('robert_frost.txt'):
    tokens = remove_punctuation(line.rstrip().lower()).split()

    T = len(tokens)
    for i in range(T):
        t = tokens[i]
        V.add(t)
        if i == 0:
            initial[t] = initial.get(t, 0. ) + 1
        else:
            t_1 = tokens[i - 1]
            if i == T - 1:
                add2dict(second_order, (t_1, t), '<END>')
            if i == 1:
                add2dict(first_order, t_1, t)
            else:
                t_2 = tokens[i - 2]
                add2dict(second_order, (t_2, t_1), t)

initial_total = sum(initial.values())
for t,c in initial.items():
    initial[t] = c / initial_total

def list2pdict(ts):
    d = {}
    n = len(ts)
    for t in ts:
        d[t] = d.get(t, 0.) + 1
    for t,c in d.items():
        d[t] = c / n
    return d

for t_1, ts in first_order.items():
    first_order[t_1] = list2pdict(ts)

for k, ts in second_order.items():
    second_order[k] = list2pdict(ts)

def sample_word(d):
    p0 = np.random.random()
    cumulative = 0
    for t,p in d.items():
        cumulative += p
        if p0 < cumulative:
            return t
    raise ValueError('Probabilities do not sum to 1.0')

def generate():
    for i in range(4):
        sentence = []

        w0 = sample_word(initial)
        sentence.append(w0)

        w1 = sample_word(first_order[w0])
        sentence.append(w1)

        while True:
            w2 = sample_word(second_order[(w0, w1)])
            if w2 == '<END>':
                break
            sentence.append(w2)
            w0, w1 = w1, w2
        print(' '.join(sentence))
generate()

V = len(V)
dense_size = V + V*V + V*V*V
sparse_size = (
    len(initial)
    + sum(len(v) for v in first_order.values())
    + sum(len(v) for v in second_order.values())
)
print(dense_size, sparse_size)
print(sparse_size / dense_size * 100)