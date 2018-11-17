import torch

from extractors.extractor import Extractor

BAD_WORDS = [
    'cock', 'piss', 'fuck', 'ass', 'shit', 'stupid', 'die', 'dick', 'moron',
    'faggot', 'crap', 'nigger', 'pussy', 'cunt', 'twat', 'bitch', 'dead', 'f*ck', 'f**k'
]


class WordCountExtractor(Extractor):
    def extract(self, text: str) -> torch.FloatTensor:
        counts = torch.zeros(len(BAD_WORDS))
        for i, word in enumerate(BAD_WORDS):
            if word in text:
                counts[i] += 1
        return counts
