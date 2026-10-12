import random

class BoundarySampler:
    '''
    Focuses on intermediate lengths and boundary lengths 0.4, 0.75, 1.1, 1.45, 1.8. 
    Gives bigger weight (trains twice as often) on the boundary lengths.
    '''
    def __init__(self, lengths = [0.4, 0.75, 1.1, 1.45, 1.8], weights = [2,1,1,1,2]):
        self.lengths = lengths
        self.weights = weights
        self.used_lengths = []
    
    def sample(self):
        '''
        Returns at random one length per sample from the predefined list if
        the length is in the used list fewer times than its weight. 
        After 7 episodes the used list is reset to empty and starts again.
        ''' 
        if len(self.used_lengths) == sum(self.weights):
            self.used_lengths = []
        length_to_train = random.choice(self.lengths)
        while self.used_lengths.count(length_to_train) >= self.weights[self.lengths.index(length_to_train)]:
            length_to_train = random.choice(self.lengths)    
        self.used_lengths.append(length_to_train)
        return length_to_train
    
    def update(self, length, steps):
        '''
        Not needed for this strategy
        '''
        pass