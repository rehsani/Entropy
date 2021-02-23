import numpy as np
np.random.seed(42)
from typing import Sequence, Any

class entropy:

    def __init__(self, sample: Sequence[Any], alpha: float, N_b: int, N_k: int):
        
        """
        Uses Quantile Spacing (QS) method for accurate probabilistic estimation 
        of one-dimensional entropy from equiprobable random samples. QS method 
        uses estimates of the quantiles that divide the support of the data 
        generating probability density function (pdf) into 
        equal-probability-mass intervals. QS requires specification of the 
        'number of quantiles' (i.e., alpha) to be used. The optimal number of 
        quantile-spacings is a fixed fraction of the sample size 
        (empirically determined to be ~0.25), and this value is insensitive to 
        distributional form or sample size (for the class of distributions 
        tested).
        
        Parameters
        --------------------------------------------------------------------------
        sample : A sequence of numbers (e.g., list or numpy.ndarray)
            Sample for which entropy is estimated
            
        alpha : float, suggested value = 0.25
            Percent of the instances from the sample used for estimation of 
            entropy (i.e., number of quantile-spacings).
            
        N_b : int, suggested value = 500
            Number of bootstraps, used to approximate the sampling variability 
            distribution of the resulting entropy estimate
            
        N_k : int, suggested value = 500
            Number of sample subsets, used to estimate the sample distribution 
            for each quantile empirically
        
        """
        
        self.sample = np.array(sample)
        self.alpha = alpha
        self.N_b = N_b
        self.N_k = N_k
    

    def estimator(self):
        """
        Estimates entropy from the 'sample' using 'alpha' percent of instances.
    
        Returns
        --------------------------------------------------------------------------
        H : float
            Estimated entropy for the sample using 'alpha' percent of instances
        """
        
        n = np.ceil(self.alpha * self.sample.size).astype(np.int64)   
        x_min = self.sample.min()
        x_max = self.sample.max()
        self.sample.sort()
        H = []
        for i in range(self.N_b):
            sample_b = np.random.choice(self.sample[1:-1], self.sample.size)
            X_alpha = [np.random.choice(sample_b[1:-1], n, replace=False) 
                       for _ in range(self.N_k)]
            X_alpha = np.vstack(X_alpha)
            X_alpha.sort(axis=1)
            Z = np.hstack([x_min, X_alpha.mean(axis=0), x_max])
            dZ = np.diff(Z)
            h = 1 / (n + 1) * np.log((n + 1) * dZ).sum()
            H.append(h)
        return np.array(H)





