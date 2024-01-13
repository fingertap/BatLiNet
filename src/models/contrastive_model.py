import torch
import random
import numpy as np
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from src.data.databundle import DataBundle, Dataset

from .nn_model import NNModel


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ContrastiveModel(NNModel):
    def build_support_dataset(self, data: Dataset, support: Dataset):
        indx = torch.randint(len(support), (len(data),), device=data.device)
        feature = support.feature[indx]
        label = support.label[indx]
        return Dataset(feature, label)

    def fit(self, dataset: DataBundle, timestamp: str):
        optimizer = optim.Adam(self.parameters())
        ori_loader = DataLoader(
            dataset.train_data, self.train_batch_size,
            shuffle=False, worker_init_fn=seed_worker)

        latest = None
        for epoch in tqdm(range(self.train_epochs), desc='Training'):
            self.train()
            sup_dataset = self.build_support_dataset(
                dataset.train_data, dataset.train_data)
            sup_loader = DataLoader(
                sup_dataset, self.train_batch_size,
                shuffle=False, worker_init_fn=seed_worker)
            for data_batch, support_batch in zip(ori_loader, sup_loader):
                x, y = data_batch.values()
                sup_x, sup_y = support_batch.values()
                optimizer.zero_grad()
                loss = self.forward(x, y, sup_x, sup_y, return_loss=True)
                loss.backward()
                optimizer.step()

            if self.workspace is not None and self.checkpoint_freq is not None\
                    and (epoch + 1) % self.checkpoint_freq == 0:
                filename = self.workspace / f'{timestamp}_epoch_{epoch+1}.ckpt'
                self.dump_checkpoint(filename)
                latest = filename

            if (epoch + 1) % self.evaluate_freq == 0:
                pred = self.predict(dataset)
                score = dataset.evaluate(pred, 'RMSE')
                print(f'[{epoch+1}/{self.train_epochs}] RMSE {score:.2f}', flush=True)

        # Create symlink latest
        if latest is not None and self.workspace is not None:
            self.link_latest_checkpoint(latest)

    @torch.no_grad()
    def predict(self, dataset: DataBundle) -> torch.Tensor:
        self.eval()
        ori_loader = DataLoader(
            dataset.test_data, self.test_batch_size,
            shuffle=False, worker_init_fn=seed_worker)
        sup_data = self.build_support_dataset(
            dataset.test_data, dataset.train_data)
        sup_loader = DataLoader(
            sup_data, self.test_batch_size,
            shuffle=False, worker_init_fn=seed_worker)
        predictions = []
        for data_batch, support_batch in zip(ori_loader, sup_loader):
            x, y = data_batch.values()
            sup_x, sup_y = support_batch.values()
            predictions.append(self.forward(x, y, sup_x, sup_y))
        predictions = torch.cat(predictions)
        return predictions
