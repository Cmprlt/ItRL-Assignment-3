'''
Pole length samplers that we can add. One class per strategy.
Each sampler has the same 2 methods which train() would call:
sample() = the pole length for the next episode
update(length,steps) = how that episode went, so how many steps survived at that length
'''
class TestSampler():
    '''
    For testing purposes before we choose the speicific strategies
    '''
    def __init__(self):
        pass
    def sample(self):
        return 1.3
    def update(self, length, steps):
        print(f"length: {length}, steps: {steps}")