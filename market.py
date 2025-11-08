from farmer import Farmer

class Market:
    def __init__(self, farmers):
        state = {
            'C' : 0,
            'W' : 0,
            'R' : 0,
            'S' : 0
        }

        for f in farmers:
            good = f.get_market()
            state[good] += 1

        self.state = state
        self.num_farmers = len(farmers)

 # 'W' : 'g1',
 # 'C' : 'g2',
 # 'S' : 'g3',
 # 'R' : 'g4'

    def wheat_price(self, q):
        return (4 + int(self.num_farmers / 5)) * 2 - q

    def corn_price(self, q):
        return 10 + int(self.num_farmers / 5) - q

    def soybeans_price(self, q):
        return self.num_farmers + 10 - int(self.num_farmers / 5) * 4 - q

    def rice_price(self, q):
        return 10 + int(self.num_farmers / 5) - q

    def get_prices(self):
        prices = {}
        prices['C'] = self.corn_price(self.state['C'])
        prices['W'] = self.wheat_price(self.state['W'])
        prices['R'] = self.rice_price(self.state['R'])
        prices['S'] = self.soybeans_price(self.state['S'])
        return prices

    def update(self, old, new):
        self.state[old] -= 1
        self.state[new] += 1
        return self.state

    def get_state(self):
        return self.state

    def summarize_round(self):
        print("prices: ")
        prices = self.get_prices()
        print(prices)
        print(self.state)
        print("profits:")
        print("Corn: ", (prices['C'] - 8) * self.state['C'])
        print("Wheat: ", (prices['W'] - 9) * self.state['W'])
        print("Rice: ", (prices['R'] - 10) * self.state['R'])
        print("Soybeans: ", (prices['S'] - 11) * self.state['S'])
        print("Sum: ", (prices['C'] - 8) * self.state['C'] + (prices['W'] - 9) * self.state['W'] + (prices['R'] - 10) * self.state['R'] + (prices['S'] - 11) * self.state['S'])
