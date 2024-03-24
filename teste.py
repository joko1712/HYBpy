import numpy as np
import torch

from sympy import symbols, diff, Matrix

import pstats

p = pstats.Stats('profile_results.prof')
p.strip_dirs().sort_stats('cumulative').print_stats(10)