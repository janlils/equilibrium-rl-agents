import numpy as np
import math
import random
from sklearn.linear_model import LinearRegression

ALL_POSSIBLE_ACTIONS = ('C', 'W', 'R', 'S')
GAMMA = 0.9
ALPHA = 0.1

np.random.seed(1411)

class Model:
    def __init__(self):
        self.theta = np.random.randn(3) + 10

    def s2x(self, s, a):
        return np.array([
            (s['C'] + 1) / sum(s.values()) if a == 'C' else 0,
            (s['W'] + 1) / sum(s.values()) if a == 'W' else 0,
            (s['R'] + 1) / sum(s.values()) if a == 'R' else 0,
            (s['S'] + 1) / sum(s.values()) if a == 'S' else 0,
            1
        ])

    def predict(self, s, a):
        x = self.s2x(s, a)
        return self.theta.dot(x)

    def grad(self, s, a):
        return self.s2x(s, a)

def getQs(model, s):
    Qs = {}
    for a in ALL_POSSIBLE_ACTIONS:
        q_sa = model.predict(s, a)
        Qs[a] = q_sa
    return Qs

def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in list(d.items()):
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

def random_action(a, eps = 0.1):
    p = np.random.random()

    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

class Farmer:
    def __init__(self, id):
        self.id = id
        self.market = np.random.choice(ALL_POSSIBLE_ACTIONS)
        self.model = Model()

        self.costs = {
            'C' : 8,
            'W' : 9,
            'R' : 10,
            'S' : 11
        }

        self.previous_state = {}
        self.next_action = ''

        self.profit = 0
        self.previous_profit = 0

    def calculate_profits(self, prices, state):
        self.previous_profit = self.profit
        self.profit = prices[self.market] - self.costs[self.market]

    def choose_action(self, state):
        s = state
        Qs = getQs(self.model, s)
        a = max_dict(Qs)[0]
        a = random_action(a)
        self.next_action = a
        self.previous_state = s

    def action(self):
        old_market = self.market
        self.market = self.next_action
        return (old_market, self.market)

    def update(self, prices, state, it):
        t =  1 + (it // 100) * 0.01
        alpha = ALPHA / t

        s = self.previous_state
        s2 = state
        r = self.profit - self.previous_profit

        old_theta = self.model.theta.copy()
        Qs2 = getQs(self.model, s2)
        a = self.next_action

        a2 = max_dict(Qs2)[0]
        a2 = random_action(a2, eps = 0.1 / t)

        self.model.theta += alpha * (r + GAMMA * self.model.predict(s2, a2) - self.model.predict(s, a))*self.model.grad(s, a)

        self.next_action = a2
        self.previous_state = s2
