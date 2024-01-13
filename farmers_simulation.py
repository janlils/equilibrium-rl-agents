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
state = market.get_state()
prices = market.get_prices()

for f in farmers:
    f.calculate_profits(prices, state)
    f.choose_action(state)

for it in range(10):
    print(it)

# ------- nowa tura -------

    # shuffle farmers
    for f in farmers:
        (old, new) = f.action()
        state = market.update(old, new)

# ------ koniec tury ---------
    state = market.get_state()
    prices = market.get_prices()

    for f in farmers:
        f.calculate_profits(prices, state)
        f.update(state)

    market.summarize_round()
