import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="colorspacious")

from . import data, utils
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
