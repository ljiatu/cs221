import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from models.kim_cnn import KimCNN


class WindowAccuracy:

    def __init__(self, model_path, loader: DataLoader):
        self.dataLoader = loader
        self.model = KimCNN(100, 6)
        self.model.load_state_dict(torch.load(model_path))

    def check_window_accuracy(self, output_path: str):
        threshold = 0.5
        with open(output_path, 'w') as output_file:
            self.model.eval()
            current_scores = defaultdict(list)
            true_output = defaultdict(list)
            with torch.no_grad():
                for text_ids, xs, ys, in self.dataLoader:
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
