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


    def get_state():
        return self.state
