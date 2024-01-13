from farmer import Farmer

class Market:
    def init(self, farmers):
        state = {
            'C' : 0,
            'W' : 0,
            'R' : 0,
            'S' : 0
        }

        for f in farmers:
            good = f.get_good()
            state[good] += 1

        self.state = state
        self.num_farmers = len(farmers)

    def corn_price(self, q):
        return 10 + int(self.num_farmers / 5) - q

    def wheat_price(self, q):
        return (4 + int(self.num_farmers) / 5) * 2 - q

    def rice_price(self, q):
        return 10 + int(self.num_farmers / 5) - q

    def soybeans_price(self, q):
        return self.num_farmers + 10 - int(self.num_farmers / 5) * 4 - q

    def get_prices(self):
        prices = {}
        prices['C'] = corn_price(self.state['C'])
        prices['W'] = wheat_price(self.state['W'])
        prices['R'] = rice_price(self.state['R'])
        prices['S'] = soybeans_price(self.state['S'])
        return prices

    def update(self, old, new):
        self.state[old] -= 1
        self.state[new] += 1

    def get_state(self):
        return self.state

    def summarize_round(self):
        print("prices: ")
        prices = self.get_prices()
        print(prices)
        print(self.state)
