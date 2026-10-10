import random

class BoundarySampler:
    '''
    Focuses on intermediate lengths and boundary lengths 0.4, 0.6, 1.2, 1.8
    '''
    def __init__(self, lengths = [0.4, 0.6, 1.2, 1.8]):
        self.lengths = lengths
        self.used_lengths = []
    
    def sample(self):
        '''
        Returns at random one length per sample from the predefined list if
        the length is not in the used list, so that each length is trained once per 4 episodes
        and the distribution is equal. After 4 episodes the used list is reset to empty and starts
        again.
        ''' 
        if len(self.used_lengths) == len(self.lengths):
            self.used_lengths = []
        length_to_train = random.choice(self.lengths)
        while length_to_train in self.used_lengths:
            length_to_train = random.choice(self.lengths)    
        self.used_lengths.append(length_to_train)
        return length_to_train
    
    def update(self, length, steps):
        '''
        Not needed for this strategy
        '''
        pass