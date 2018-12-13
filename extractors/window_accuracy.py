from collections import defaultdict

import torch
from torch.utils.data import DataLoader


class WindowAccuracy:
    def __init__(self, model_path: str, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.model = torch.load(model_path).to(device=device)

    def check_window_accuracy(self, output_path: str):
        threshold = 0.5
        with open(output_path, 'w') as output_file:
            self.model.eval()
            current_scores = defaultdict(list)
            true_output = defaultdict(list)
            with torch.no_grad():
                for text_ids, xs, ys in self.loader:
                    xs = xs.to(device=self.device)
                    ys = ys.to(device=self.device)
                    raw_scores = self.model(xs)
                    probabilities = torch.sigmoid(raw_scores)
                    for idx, text_id in enumerate(text_ids):
                        true_output[text_id] = ys[idx]
                        output = [score > threshold for score in probabilities[idx]]
                        old_scores = current_scores[text_id]
                        if len(old_scores) > 0:
                            current_scores[text_id] = [output[i] or old_scores[i] for i in range(len(output))]
                        else:
                            current_scores[text_id] = output

            num_samples = len(current_scores)
            num_correct = 0
            for k, v in current_scores.items():
                if v == true_output[k]:
                    num_correct += 1
            print(f'Samples: {num_samples}, Num correct: {num_correct}, Accuracy: {num_correct / num_samples}')
