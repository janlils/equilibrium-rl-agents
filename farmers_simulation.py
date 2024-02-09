"""
Choose apropriate Farmer and Market type (with or without government subsidies)
"""
# from farmer import Farmer
# from market import Market

from farmer_gov import Farmer
from market_gov import Market

import pickle
import numpy as np
import math
import pandas as pd
from random import shuffle

np.random.seed(1411)
NUM_OF_FARMERS = 100

#agents_market, agents_profit for analysis/visualization purposes
agents_market = []
agents_profit = []

# repeat whole experiment 100 times
for i in range(100):
    # print(i)

    # create farmers
    farmers = []

    for id in range(NUM_OF_FARMERS):
        farmers.append(Farmer(id))

    # initialize market, state, prices
    market = Market(farmers)
    state = market.get_state()
    prices = market.get_prices()

    # profits and markets for analysis/visualization purposes
    # first two elements are: experiment_id and iteration_id
    profits = [i, 0]
    markets = [i, 0]
    for f in farmers:
        profits.append(f.calculate_profits(prices, state))
        markets.append(f.get_market())

    agents_market.append(markets)
    agents_profit.append(profits)

    for it in range(1000):
    # ------- start round -------
        profits = [i, it + 1] + [0] * NUM_OF_FARMERS
        markets = [i, it + 1] + [''] * NUM_OF_FARMERS

        shuffle(farmers)
        for f in farmers:
            f.choose_action(state, it)
            (old, new) = f.action() #execute action, return old market and new market
            state = market.update(old, new)

    # ------ end round ---------

        # calcualte profits and update farmers
        state = market.get_state()
        prices = market.get_prices()

        for f in farmers:
            p = f.calculate_profits(prices, state)
            f.update(prices, state, it)

            f_id = f.get_id() + 2
            markets[f_id] = f.get_market()
            profits[f_id] = p

        agents_market.append(markets)
        agents_profit.append(profits)

columns = ['Round', 'Iteration'] + list(range(NUM_OF_FARMERS))
results_market = pd.DataFrame(agents_market, columns = columns)
results_profits = pd.DataFrame(agents_profit, columns = columns)

print(results_market)
print(results_profits)

with open('results_market_gov.pickle', 'wb') as f:
    pickle.dump(results_market, f)

with open('results_profits_gov.pickle', 'wb') as f:
    pickle.dump(results_profits, f)
