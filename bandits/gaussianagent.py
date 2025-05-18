import numpy as np

class GaussianSamplingAgent :
    def __init__(self, k, mu_prior=0.0, tau_prior=1.0, sigma=1.0) :
        self.k = k
        self.mu = np.zeros(k) # Posterior Mean
        self.lambda_ = np.ones(k) # Precision = 1 / Variance
        self.N = np.zeros(k)
        self.mu_prior = mu_prior
        self.tau_prior = tau_prior
        self.sigma = sigma
    
    def select_action(self) :
        stddev = np.sqrt(1 / self.lambda_)
        samples = np.random.normal(self.mu, stddev)
        return np.argmax(samples)
    
    def update(self, action: int, reward: float) -> int :
        self.N[action] += 1
        self.lambda_[action] += (1 / (self.sigma ** 2))
        self.mu[action] = (
            (self.lambda_[action] - 1) * self.mu[action] + reward / (self.sigma ** 2)
        ) / self.lambda_[action]