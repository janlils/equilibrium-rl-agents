

from farmer_gov import Farmer
from market_gov import Market

import numpy as np
import math
import pandas as pd
from random import shuffle

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

for it in range(10000):

# ------- nowa tura -------
    shuffle(farmers)
    for f in farmers:
        f.choose_action(state, it)
        (old, new) = f.action()
        state = market.update(old, new)

# ------ koniec tury ---------
    state = market.get_state()
    prices = market.get_prices()

    for f in farmers:
        f.calculate_profits(prices, state)
        f.update(prices, state, it)

    if it % 100 == 0 or it > 9950:
        market.summarize_round()
        # if it > 9995:
        #     for f in farmers:
        #         f.print_stats()
