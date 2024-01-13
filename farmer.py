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
            math.sqrt((s[0] + 1)*(s[1])) - math.sqrt((s[0] * s[1])) if a == 'fish' else 0,
            math.sqrt((s[0])*(s[1] + 1)) - math.sqrt((s[0] * s[1])) if a == 'coconut' else 0,
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

        s = self.get_state()
        Qs = getQs(self.model, s)
        a = max_dict(Qs)[0]
        a = random_action(a)
        self.next_action = a
        self.previous_state = self.get_state()
        self.profit = 0
        self.previous_profit = 0
        self.money

    def produce_random(self):
        p = np.random.random()
        if p > 0.5:
            self.create('fish', self.fish_prod_function)
        else:
            self.create('coconut', self.coconut_prod_function)

    def produce(self):
        good = self.next_action
        self.previous_state = self.get_state()
        p1 = self.get_profit()
        self.previous_profit= p1
        if good == 'fish':
            self.create('fish', self.fish_prod_function)
        elif good == 'coconut':
            self.create('coconut', self.coconut_prod_function)
        else:
            print("Can't produce: " + good)
        return self.get_utility() - u1

    def get_utility(self):
        return self.utility_function(**{'fish': self['fish'], 'coconut' : self['coconut']})

    # def print_utility(self):
    #     fishes = self['fish']
    #     coconuts = self['coconut']
    #     utility = self.consume(self.utility_function, self.consume_everything)
    #     self.create('fish', fishes)
    #     self.create('coconut', coconuts)
    #     fish_u = self.params[0]['fish']
    #     coconut_u = self.params[0]['coconut']
    #     fish_prod = self.params[1][0]
    #     coconut_prod = self.params[1][1]
    #     return [self.id, utility, fishes, coconuts, self.price, self.num_trades,
    #             self.fish_produced, self.coconut_produced, fish_u, coconut_u, fish_prod, coconut_prod]

    def get_state(self):
        return (self['fish'], self['coconut'])

    def get_state_id(self):
        return (self.id, self.get_state(), self.model.theta, self.price)

    def finalize_round(self, it, time):
        t =  1 + (it // 100) * 0.01
        alpha = ALPHA / t

        s = self.previous_state
        s2 = self.get_state()
        r = self.get_utility() - self.previous_utility
        self.adjust_price()

        old_theta = self.model.theta.copy()
        Qs2 = getQs(self.model, s2)
        a = self.next_action
        a2 = max_dict(Qs2)[0]
        a2 = random_action(a2, eps = 0.1 / t)
        self.model.theta += alpha * (r + GAMMA * self.model.predict(s2, a2) - self.model.predict(s, a))*self.model.grad(s, a)
        self.next_action = a2
        self.previous_state = s2
        self.previous_utility = self.get_utility()
        # if time == 350:
        #     print(self.id, self.trade_try, self.num_trades, self.price)
