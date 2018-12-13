from datetime import datetime
import time

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
            self,
            model: Module,
            loss_func: Module,
            optimizer: Optimizer,
            device: torch.device,
            model_type: str,
            loader_train: DataLoader,
            loader_val: DataLoader,
            loader_test: DataLoader,
            num_epochs: int = 10,
            print_every: int = 50,
    ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device
        self.model_type = model_type
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

            for t, (_, x, y) in enumerate(self.loader_train):
                self.model.train()
                x = x.to(device=self.device)
                y = y.to(device=self.device)

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
        print('Saving model...')
        torch.save(self.model, f'saved_models/{self.model_type}_{datetime.now().isoformat()}.model')
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
            for _, x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                raw_scores = self.model(x)
                loss = self.loss_func(raw_scores, y)
                total_loss += loss.item()
                total_num_samples += x.size(0)

            total_loss /= total_num_samples
            print(f'{loader_label.capitalize()} Loss: {total_loss}')
            return total_loss

    def _write_results(self, output_path: str):
        with open(output_path, 'w') as output_file:
            output_file.write('id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n')
            with torch.no_grad():
                for text_ids, x, _ in self.loader_test:
                    x = x.to(device=self.device)
                    raw_scores = self.model(x)
                    probabilities = torch.sigmoid(raw_scores)
                    for idx, text_id in enumerate(text_ids):
                        probabilities_csv = ','.join([str(prob.item()) for prob in probabilities[idx]])
                        output_file.write(f'{text_id},{probabilities_csv}\n')
