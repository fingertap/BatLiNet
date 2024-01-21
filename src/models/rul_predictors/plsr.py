from sklearn.cross_decomposition import PLSRegression

from src.builders import MODELS
from src.models.sklearn_model import SkleanModel


@MODELS.register()
class PLSRRULPredictor(SkleanModel):
    def __init__(self, *args, workspace: str = None, seed: int = 0, **kwargs):
        SkleanModel.__init__(self, workspace, seed)
        self.model = PLSRegression(*args, **kwargs)
