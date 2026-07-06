import random
import re
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
pattern = re.compile('[^a-zA-Z]')

def generate_cipher():
    return ''.join(random.sample(ALPHABET, len(ALPHABET)))

def encode_message(msg, cipher):
    msg = pattern.sub(' ', msg.lower())
    coded_msg = []
    for ch in msg:
        coded_ch = ch
        if ch in cipher:
            i = ALPHABET.index(ch)
            coded_ch = cipher[i]
        coded_msg.append(coded_ch)
    return ''.join(coded_msg)

def decode_message(msg, cipher):
    decoded_msg = []
    for ch in msg:
        decoded_ch = ch
        if ch in cipher:
            i = cipher.index(ch)
            decoded_ch = ALPHABET[i]
        decoded_msg.append(decoded_ch)
    return ''.join(decoded_msg)