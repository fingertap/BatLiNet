from sklearn.linear_model import ElasticNetCV

from src.builders import MODELS
from src.models.sklearn_model import SkleanModel


@MODELS.register()
class ElasticNetRULPredictor(SkleanModel):
    def __init__(self, *args, workspace: str = None, seed: int = 0, **kwargs):
        SkleanModel.__init__(self, workspace, seed)
        self.model = ElasticNetCV(**kwargs)
