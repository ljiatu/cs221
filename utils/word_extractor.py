import torch

BAD_WORDS = [
    'cock', 'piss', 'fuck', 'ass', 'shit', 'stupid', 'die', 'dick', 'moron',
    'faggot', 'crap', 'nigger', 'pussy', 'cunt', 'twat', 'bitch', 'dead', 'f*ck', 'f**k'
]


def extract(text):
    counts = torch.zeros(len(BAD_WORDS))
    for i, word in enumerate(BAD_WORDS):
        if word in text:
            counts[i] += 1
    return counts
