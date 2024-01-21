from sklearn.ensemble import RandomForestRegressor

from src.builders import MODELS
from src.models.sklearn_model import SkleanModel


@MODELS.register()
class RandomForestRULPredictor(SkleanModel):
    def __init__(self, *args, workspace: str = None, seed: int = 0, **kwargs):
        SkleanModel.__init__(self, workspace, seed)
        self.model = RandomForestRegressor(*args, **kwargs)
