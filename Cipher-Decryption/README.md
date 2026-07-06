Genetic Cipher Decoder

This repository implements a genetic algorithm-based solver for simple substitution ciphers using a Markov language model. The algorithm evolves candidate cipher permutations (DNA) to maximize the log-likelihood of decoded messages according to the learned letter transition probabilities from a training corpus (e.g., Moby Dick).

Features

Learns letter transition probabilities from a text corpus.

Encodes and decodes messages using random substitution ciphers.

Uses a genetic algorithm with swaps and selection to iteratively improve cipher guesses.

Visualizes the evolution of average log-likelihood over iterations.

Works well for messages similar to the training text and can be adapted for general text.

Usage

Train the language model on a text file.

Encode a message using a random cipher.

Run the genetic algorithm to decode the message.

Inspect decoded results and evolutionary progress.
<img width="638" height="548" alt="image" src="https://github.com/user-attachments/assets/b036d741-8f9c-48c4-8f5e-11144d15bff7" />
```
Iteration 0, best score: -567.219554004821, avg score: -764.306721677915
Iteration 200, best score: -357.50771030398323, avg score: -468.1858348811088
Iteration 400, best score: -321.7773680828188, avg score: -441.0619512048761
Iteration 600, best score: -321.7773680828188, avg score: -446.78036230767077
Iteration 800, best score: -321.7773680828188, avg score: -442.9357290468413
Decoded message: i cas on the foint ow asping him chat that corp might be but something in his manner shoced me that the question could be an uncelyome one  i fondered oker our short yonkersation 
Original message: I was on the point of asking him what that work might be but something in his manner showed me that the question would be an unwelcome one. I pondered over our short conversation.
Log Likelihood of decoded message: -321.7773680828188
LL of original message: -327.3294313515325
c -> encoded: o, decoded: s
f -> encoded: u, decoded: z
k -> encoded: j, decoded: m
p -> encoded: z, decoded: j
v -> encoded: m, decoded: x
w -> encoded: s, decoded: u
y -> encoded: d, decoded: o
z -> encoded: x, decoded: d
```
