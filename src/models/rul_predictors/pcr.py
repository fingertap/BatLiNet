from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from src.builders import MODELS
from src.models.sklearn_model import SkleanModel


@MODELS.register()
class PCRRULPredictor(SkleanModel):
    def __init__(self, *args, workspace: str = None, seed: int = 0, **kwargs):
        SkleanModel.__init__(self, workspace, seed)
        self.model = make_pipeline(PCA(*args, **kwargs), LinearRegression())
