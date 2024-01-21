from sklearn.svm import SVR

from src.builders import MODELS
from src.models.sklearn_model import SkleanModel


@MODELS.register()
class SVMRULPredictor(SkleanModel):
    def __init__(self, *args, workspace: str = None, seed: int = 0, **kwargs):
        SkleanModel.__init__(self, workspace, seed)
        self.model = SVR(*args, **kwargs)
