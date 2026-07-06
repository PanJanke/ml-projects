import numpy as np

class LanguageModel:
    def __init__(self, alphabet_size=26):
        self.alphabet_size = alphabet_size
        self.M = np.ones((alphabet_size, alphabet_size))
        self.pi = np.zeros(alphabet_size)

    def _char_to_index(self, ch):
        return ord(ch) - 97

    def update_transition(self, ch1, ch2):
        i = self._char_to_index(ch1)
        j = self._char_to_index(ch2)
        self.M[i, j] += 1

    def update_pi(self, ch):
        i = self._char_to_index(ch)
        self.pi[i] += 1

    def get_word_prob(self, word):
        i = self._char_to_index(word[0])
        logp = np.log(self.pi[i])

        for ch in word[1:]:
            j = self._char_to_index(ch)
            logp += np.log(self.M[i, j])
            i = j
        return logp

    def get_sequence_prob(self, words):
        if isinstance(words, str):
            words = words.split()

        logp = 0
        for word in words:
            logp += self.get_word_prob(word)
        return logp

    def normalize(self):
        self.pi = self.pi / np.sum(self.pi)
        self.M = self.M / np.sum(self.M, axis=1, keepdims=True)