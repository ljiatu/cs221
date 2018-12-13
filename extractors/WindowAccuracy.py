from torch.nn import Module
import torch
from torch.utils.data import DataLoader
from collections import defaultdict


class WindowAccuracy:

    def __init__(self, model_path, loader: DataLoader):
        self.dataLoader = loader
        self.model = Module()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def check_window_accuracy(self, output_path: str) -> float:
        threshold = 0.5
        with open(output_path, 'w') as output_file:
            self.model.eval()
            current_scores = defaultdict(list)
            true_output = defaultdict(list)
            with torch.no_grad():
                for text_id, x, y, in self.loader:
                    raw_scores = self.model(x)
                    true_output[text_id] = y
                    probabilities = torch.sigmoid(raw_scores[0])
                    output = [int(score > threshold) for score in probabilities]
                    old_scores = current_scores[text_id]
                    if len(old_scores) > 0:
                        current_scores[text_id] = [int(output[i] or old_scores[i]) for i in range(len(output))]
                    else:
                        current_scores[text_id] = output
                    # output_file.write(f'(Window) Text id: {text_id}, y: {y}, Prob: {probabilities}\n')
            num_samples = len(current_scores)
            num_correct = 0
            for k, v in current_scores.items():
                prediction = [int(i >= 1) for i in v]
                if prediction == true_output[k]:
                    num_correct += 1
            print(f'Samples: {num_samples}, Num correct: {num_correct}, Accuracy: {float(num_correct) / num_samples}')
            # output_file.write(f'(Final) Text id: {k}, Output: {prediction}\n')
