from sklearn.linear_model import Ridge

from src.builders import MODELS
from src.models.sklearn_model import SkleanModel


@MODELS.register()
class RidgeRULPredictor(SkleanModel):
    def __init__(self, *args, workspace: str = None, **kwargs):
        SkleanModel.__init__(self, workspace)
        self.model = Ridge(**kwargs)
