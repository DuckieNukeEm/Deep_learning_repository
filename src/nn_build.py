import pandas as pd
import numpy as np
from typing import Union
from tensorflow.keras.utils import Sequence, to_categorical # For our own data generator
import matplotlib.pyplot as plt # for showing the val vs train model
from tensorflow.keras.callbacks import ModelCheckpoint


class nn_model():
    """simple class to easily build models"""