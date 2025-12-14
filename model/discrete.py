import numpy as np


class GlazeDiscrete:
    """
    Minimal discrete Glaze-style belief updater.

    Latent state x_t ∈ {0, 1}
    Observations o_t ∈ {0, 1}
    """

    def __init__(self, hazard=0.05, obs_noise=0.1):
        """
        hazard     : probability of latent state switch
        obs_noise  : P(o_t != x_t)
        """
        self.h = hazard
        self.eps = obs_noise

        # log-odds belief: log P(x=1) / P(x=0)
        self.L = 0.0

    def reset(self):
        self.L = 0.0

    def update(self, observation):
        """
        observation ∈ {0,1}
        returns posterior log-odds
        """

        # ----- 1. hazard-based leak (prior update) -----
        # P(x_t) = (1-h)*P(x_{t-1}) + h*(1-P(x_{t-1}))
        p_prev = 1 / (1 + np.exp(-self.L))
        p_pred = (1 - self.h) * p_prev + self.h * (1 - p_prev)

        L_pred = np.log(p_pred / (1 - p_pred))

        # ----- 2. evidence update (likelihood) -----
        if observation == 1:
            llr = np.log((1 - self.eps) / self.eps)
        else:
            llr = np.log(self.eps / (1 - self.eps))

        self.L = L_pred + llr
        return self.L

    def mean_belief(self):
        """Return P(x=1)"""
        return 1 / (1 + np.exp(-self.L))
