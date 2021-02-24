import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

import lda.datasets  # 使用第三方的lda库，安装方式: pip install lda
from pprint import pprint

lda.LDA(n_topics=topic_num, n_iter=500, random_state=28)