# Entropy Estimation Using Quantile Spacing Approach

We have developed a simple Quantile Spacing (QS) method for accurate probabilistic estimation of one-dimensional entropy from equiprobable random samples. In contrast to Bin Counting (BC) method, which uses equal-width bins with varying probability mass, the QS method uses estimates of the quantiles that divide the support of the data generating probability density function (pdf) into equal-probability-mass intervals. Whereas BC requires optimal tuning of a bin-width hyper-parameter whose value varies with sample size and shape of the pdf, QS requires specification of the number of quantiles to be used. For the class of distributions tested, that the optimal number of quantile-spacings is a fixed fraction of the sample size (empirically determined to be ~0.25-0.35), and that this value is relatively insensitive to distributional form or sample size, providing a clear advantage over BC since hyperparameter tuning is not required. Bootstrapping is used to approximate the sampling variability distribution of the resulting entropy estimate, and is shown to accurately reflect the true uncertainty. For the four distributional forms studied (Gaussian, Log-Normal, Exponential and Bimodal Gaussian Mixture), expected estimation bias is less than 1% and uncertainty is relatively low even for very small sample sizes. We speculate that estimating quantile locations, rather than bin-probabilities, results in more efficient use of the information in the data to approximate the underlying shape of an unknown data generating pdf.

For more information please see the [manuscript](https://arxiv.org/abs/2102.12675). If you have any question, feel free to contact us at rehsani@email.arizona.edu or hoshin@email.arizona.edu.

First, we need to import some required libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
```
## Example 1 - Python

Put the entropy.py file in the directory you are working with and import it as a library:

```python
import entropy
```

Here we use a sample of size 5,000 from a Guassian distribution (μ=0, σ=1) with known true entropy (H=1.4187...) to test the algorithm:

```python
mu = 0
sigma = 1
H_true = 0.5 * np.log(2*np.pi*np.exp(1)*sigma**2)

n = 5000
sample = np.random.normal(mu, sigma, n)
H = entropy(sample, alpha=0.25, N_b=100, N_k=500)
H = H.estimator()
```

Let's take a look at the estimated entropy:

```python
plt.boxplot(H)
plt.ylabel('Estimated Entropy')
plt.text(0.05, 0.9,
         'True Entropy = {} \nMean Estimated Entropy = {} \n'.
         format(H_true.round(3), H.mean().round(3)),
         horizontalalignment='left',
         verticalalignment='center',
         transform = plt.gca().transAxes)
```

![](https://github.com/rehsani/Entropy/blob/master/Example1.png)

## Example 2 - MATLAB

Put the entropy.m file in the directory you are working with.
Here we use a sample of size 5,000 from a Guassian distribution (μ=0, σ=1) with known true entropy (H=1.4189...) to test the algorithm:

```matlab
mu = 0;
sigma = 1;
H_true = 0.5 * log(2 * pi * exp(1) * sigma ^ 2);

n = 5000;
sample = normrnd(mu, sigma, 1, n);
H = entropy(sample, 0.25, 100, 500);
```

Let's take a look at the estimated entropy:

```python
boxplot(H);
xticks([]);
ylabel('Estimated Entropy');
txt = 'True Entropy = %.3f\nMean Estimated Entropy = %.3f';
text(0.05, 0.9, sprintf(txt, H_true, mean(H)), 'Units','normalized');
```

![](https://github.com/rehsani/Entropy/blob/master/Example2.png)
