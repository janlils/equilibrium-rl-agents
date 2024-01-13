from farmer import Farmer
from market import Market

import numpy as np
import math
import pandas as pd

np.random.seed(1411)
NUM_OF_FARMERS = 100

farmers = []

for id in range(NUM_OF_FARMERS):
    farmers.append(Farmer(id))

market = Market(farmers)

for it in range(10):
    print(it)
    state = market.get_state()
    prices = market.set_prices()
    
    farmers.calculate_profits(prices, state)
    farmers.choose_market(state)
    market.summarize_round()
