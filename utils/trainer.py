import time

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from nltk import word_tokenize

class Trainer:
    def __init__(
            self,
            model: Module,
            loss_func: Module,
            optimizer: Optimizer,
            loader_train: DataLoader,
            loader_val: DataLoader,
            loader_test: DataLoader,
            num_epochs: int = 10,
            print_every: int = 50,
    ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.num_epochs = num_epochs
        self.print_every = print_every

    def train(self):
        start = time.time()

        for e in range(self.num_epochs):
            print('-' * 10)
            print(f'Epoch {e}')
            print('-' * 10)

            running_loss = 0.0
            total_samples = 0

            for t, (text_id, x, y) in enumerate(self.loader_train):
                self.model.train()

                raw_scores = self.model(x)
                loss = self.loss_func(raw_scores, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Keep track of training loss throughout the epoch.
                training_loss = loss.item() * x.size(0)
                running_loss += training_loss
                total_samples += x.size(0)

                if t % self.print_every == 0:
                    print('Iteration %d, training loss = %.4f' % (t, loss.item()))
                    self._check_accuracy('validation', self.loader_val)
                    print()

            epoch_training_loss = running_loss / total_samples
            epoch_val_loss = self._check_accuracy('validation', self.loader_val)
            print('*' * 30)
            print(f'End of epoch {e} summary')
            print(f'Total samples: {total_samples}')
            print(f'Average training loss: {epoch_training_loss}')
            print(f'Val loss: {epoch_val_loss}')
            print('*' * 30)

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def test(self, output_path: str):
        print('Testing...')
        self.model.eval()
        self._check_accuracy('test', self.loader_test)
        self._write_results(output_path)

    def _check_accuracy(self, loader_label: str, loader: DataLoader) -> float:
        total_num_samples = 0
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for text_id, x, y in loader:
                raw_scores = self.model(x)
                loss = self.loss_func(raw_scores, y)
                total_loss += loss.item()
                total_num_samples += x.size(0)

            total_loss /= total_num_samples
            print(f'{loader_label.capitalize()} Loss: {total_loss}')
            return total_loss

    def check_windows(self, loader: DataLoader, num_samples: int, output_path: str):
        with open(output_path, 'w') as output_file:
            self.model.eval()
            samples_read = 0
            with torch.no_grad():
                for text_id, x, y in loader:
                    raw_scores = self.model(x)
                    output_file.write(f'Text id:{text_id}, Raw Scores:{raw_scores}, True Scores:{y}\n')
                    samples_read += 1
                    if samples_read >= num_samples:
                        break
            output_file.close()

    def _write_results(self, output_path: str):
        with open(output_path, 'w') as output_file:
            output_file.write('id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n')
            with torch.no_grad():
                for text_id, x, y in self.loader_test:
                    raw_scores = self.model(x)
                    probabilities = torch.sigmoid(raw_scores[0])
                    probabilities_csv = ','.join([str(prob.item()) for prob in probabilities])
                    output_file.write(f'{text_id[0]},{probabilities_csv}\n')
