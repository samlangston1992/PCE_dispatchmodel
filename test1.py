import pandas as pd
from datetime import timedelta
from battery_optimisation_function import battery_optimisation
from tqdm import tqdm
from datetime import timedelta
import os

from pyomo.environ import *
from pyutilib.services import register_executable, registered_executable

register_executable(name='glpsol')