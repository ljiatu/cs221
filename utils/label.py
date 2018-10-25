import torch

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


class Label:
    def __init__(
            self,
            toxic: str,
            severe_toxic: str,
            obscene: str,
            threat: str,
            insult: str,
            identity_hate: str,
    ):
        self._exists = [
            self._to_bool(toxic),
            self._to_bool(severe_toxic),
            self._to_bool(obscene),
            self._to_bool(threat),
            self._to_bool(insult),
            self._to_bool(identity_hate),
        ]

    def __repr__(self):
        return ','.join([label for label, exists in zip(LABELS, self._exists) if exists])

    def tensor(self):
        return torch.tensor(self._exists)

    @staticmethod
    def _to_bool(indicator: str):
        return bool(int(indicator))
