import numpy as np
class Classifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
        self.K = len(logpriors) # liczba klas

    def compute_log_likehood(self, input_, class_):
        logA = self.logAs[class_]
        logpi = self.logpis[class_]

        last_idx = None
        logprob = 0.0
        for idx in input_:
            if last_idx is None:
                logprob += logpi[idx]
            else:
                logprob += logA[last_idx, idx]

            last_idx = idx
        return logprob

    def predict(self, inputs):
        predictions = np.zeros(len(inputs))
        for i, input_ in enumerate(inputs):
            posteriors = [self.compute_log_likehood(input_, c)+self.logpriors[c] for c in range(self.K)]
            pred = np.argmax(posteriors)
            predictions[i] = pred
        return predictions