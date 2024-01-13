import abc
import torch
import pickle

from src.data.databundle import DataBundle

from .base import BaseModel


class SkleanModel(BaseModel, abc.ABC):
    def fit(self, dataset: DataBundle, timestamp: str = None) -> None:
        device = dataset.device
        dataset = dataset.to('cpu')
        feature = dataset.train_data.feature
        feature = feature.view(len(feature), -1)
        self.model.fit(feature, dataset.train_data.label.to('cpu'))

        timestamp = timestamp or 'UnknownTime'

        # Dump models
        if self.workspace is not None:
            filename = self.workspace / f'{timestamp}.ckpt'
            self.dump_checkpoint(filename)
            self.link_latest_checkpoint(filename)

        dataset = dataset.to(device)

    def predict(self, dataset: DataBundle) -> torch.Tensor:
        device = dataset.device
        dataset = dataset.to('cpu')

        feature = dataset.test_data.feature
        feature = feature.view(len(feature), -1)
        scores = self.model.predict(feature.numpy())

        scores = torch.from_numpy(scores).to(device).view(-1)
        dataset = dataset.to(device)
        return scores

    def dump_checkpoint(self, path: str):
        with open(path, 'wb') as fout:
            pickle.dump(self.model, fout)

    def load_checkpoint(self, path: str):
        with open(path, 'rb') as fin:
            self.model = pickle.load(fin)
