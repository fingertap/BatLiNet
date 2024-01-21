from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct

from src.builders import MODELS
from src.models.sklearn_model import SkleanModel


@MODELS.register()
class GaussianProcessRULPredictor(SkleanModel):
    def __init__(self, *args, workspace: str = None, seed: int = 0, **kwargs):
        SkleanModel.__init__(self, workspace, seed)
        kernel = DotProduct() + RBF()
        self.model = GaussianProcessRegressor(kernel)
