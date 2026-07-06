import numpy as np
import re
from language_model import LanguageModel
from cipher_utils import generate_cipher, encode_message, decode_message

# 1. Nauka modelu językowego

lm = LanguageModel()
pattern = re.compile('[^a-zA-Z]')

with open('mobydick.txt') as f:
    for line in f:
        line = line.rstrip()
        line = pattern.sub(' ', line).lower()
        if line:
            tokens = line.split()
            for token in tokens:
                ch0 = token[0]
                lm.update_pi(ch0)
                for ch1 in token[1:]:
                    lm.update_transition(ch0, ch1)
                    ch0 = ch1
lm.normalize()

# 2. Przygotowanie szyfru i wiadomości
#original_message = "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people’s hats off—then, I account it high time to get to sea as soon as I can."
#original_message = "The harpoon was darted; the stricken whale flew forward; with igniting velocity the line ran through the grooves ran foul. Ahab stooped to clear it; he did clear it; but the flying turn caught him round the neck, and voicelessly as Turkish mutes bowstring their victim, he was shot out of the boat, ere the crew knew he was gone. Next instant, the heavy eye-splice in the rope’s final end flew out of the stark-empty."
original_message = "I was on the point of asking him what that work might be but something in his manner showed me that the question would be an unwelcome one. I pondered over our short conversation."
cipher = generate_cipher()
encoded_message = encode_message(original_message, cipher)

# 3. Inicjalizacja populacji DNA
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
dna_pool_size = 200
n_children = 10
num_of_keeped = 20
num_iters = 1000
dna_pool = [''.join(np.random.permutation(list(ALPHABET))) for _ in range(dna_pool_size)]
scores = np.zeros(num_iters)
best_dna = None
best_score = float('-inf')

def evolve_offspring(dna_pool, n_children, n_swaps=3):
    offspring = []
    for dna in dna_pool:
        for _ in range(n_children):
            copy = list(dna)
            for _ in range(n_swaps):
                j = np.random.randint(len(copy))
                k = np.random.randint(len(copy))
                copy[j], copy[k] = copy[k], copy[j]
            offspring.append(''.join(copy))
    return offspring + dna_pool

# 4. Pętla ewolucyjna
for i in range(num_iters):
    if i > 0:
        dna_pool = evolve_offspring(dna_pool, n_children)

    dna2score = {}
    for dna in dna_pool:
        decoded = decode_message(encoded_message, dna)
        score = lm.get_sequence_prob(decoded)
        dna2score[dna] = score
        if score > best_score:
            best_score = score
            best_dna = dna

    # keep top DNA
    sorted_dna = sorted(dna2score.items(), key=lambda x: x[1], reverse=True)
    dna_pool = [dna for dna, _ in sorted_dna[:num_of_keeped]]

    scores[i] = np.mean(list(dna2score.values()))

    if i % 200 == 0:
        print(f"Iteration {i}, best score: {best_score}, avg score: {scores[i]}")

# 5. Wyniki

decoded_message = decode_message(encoded_message, best_dna)
print("Decoded message:", decoded_message)
print("Original message:", original_message)
print("Log Likelihood of decoded message:", lm.get_sequence_prob(decoded_message))
print("LL of original message:", lm.get_sequence_prob(pattern.sub(' ', original_message).lower()))

# które litery są niepoprawnie odgadnięte
for i, letter in enumerate(ALPHABET):
    if cipher[i] != best_dna[i]:
        print(f"{letter} -> encoded: {cipher[i]}, decoded: {best_dna[i]}")

import matplotlib.pyplot as plt
plt.plot(scores)
plt.show()