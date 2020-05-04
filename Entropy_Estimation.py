from numpy import array, copy, concatenate, sum, log, append
from numpy.random import choice
from statistics import mean
from pandas import DataFrame
    
class Entropy:
    
    alpha = array([1, 2, 5, 10, 20, 50])
    
    #Initializer / Sample Attributes
    def __init__(self, sample, population_min=None, population_max=None,
                 N_k=100, N_x=100):
        
        self.sample = array(sample)
        self.population_min= population_min
        self.population_max = population_max
        self.N_k = N_k
        self.N_x = N_x
    
    def sample_sorter(self):
        
        sorted_sample = self.sample.copy()
        sorted_sample.sort()
        self.sorted_sample = sorted_sample
        
    def sample_MinMax(self):
        
        delta = self.sorted_sample[1:] - self.sorted_sample[:-1]
        
        if self.population_min == None and self.population_max == None: 
            
            x_min = [self.sorted_sample[0] - delta[0]]
            x_max = [self.sorted_sample[-1] + delta [-1]]
            extended_sample = concatenate([x_min, self.sorted_sample, x_max])
            
        elif self.population_min != None and self.population_max != None: 
            
            #CHECK X_MIN and X_MAX
            x_min = [self.population_min]
            x_max = [self.population_max]
            extended_sample = concatenate([x_min, self.sorted_sample, x_max])
            
        elif self.population_min != None and self.population_max == None:
            
             x_min = [self.population_min]
             x_max = [self.sorted_sample[-1] + delta [-1]]
             extended_sample = concatenate([x_min, self.sorted_sample, x_max])
             
        elif self.population_min == None and self.population_max != None:  
            
            x_min = [self.sorted_sample[0] - delta[0]]
            x_max = [self.population_max]
            extended_sample = concatenate([x_min, self.sorted_sample, x_max])
            
        self.extended_sample = extended_sample
    
    def sampler(self, alpha):
        
        sub_sample_size = int(len(self.sample) * alpha / 100)
        sub_sample = choice(self.sample, sub_sample_size, False)
        return sub_sample
    
    def estimator(sub_sample):
    
        delta = sub_sample[1:] - sub_sample[:-1]
        N_int = len(sub_sample)
        H = (1/N_int) * sum(log(N_int * delta))
        return H

    def call(self):
        self.sample_sorter()
        self.sample_MinMax()
        sub_samples = {i:[] for i in self.alpha}
        Z = {i:[] for i in self.alpha}
        H = {i:[] for i in self.alpha}
        for i in self.alpha:
            for j in range(self.N_x):
                dummy = []
                for _ in range (self.N_k):
                    dummy.append(self.sampler(i))
                dummy = array(dummy)
                sub_samples[i].append(dummy)
                dummy.sort(axis=1)
                Z_bar = dummy.mean(axis=0)
                Z[i].append(Z_bar)
                H[i].append(Entropy.estimator(Z_bar))
            sub_samples[i] = array(sub_samples[i])
            Z[i] = array(Z[i])
            H[i] = array(H[i])
        self.sub_samples = sub_samples
        self.Z = Z
        self.estimated_entropy = H
        # df = DataFrame(H)
        # df.plot.box(grid=True)
        
        
        
        
        
        
    