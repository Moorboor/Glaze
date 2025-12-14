class GlazeContinuous:
    """
    Minimal continuous Glaze-style belief updater.

    Latent state x_t ∈ ℝ
    Observations o_t ~ N(x_t, obs_var)
    """

    def __init__(self, hazard=0.05, obs_var=1.0, reset_var=10.0):
        """
        hazard    : probability of change point
        obs_var   : observation noise variance
        reset_var : variance after a change
        """
        self.h = hazard
        self.obs_var = obs_var
        self.reset_var = reset_var

        # belief state
        self.mu = 0.0
        self.var = reset_var

    def reset(self):
        self.mu = 0.0
        self.var = self.reset_var

    def update(self, observation):
        """
        observation ∈ ℝ
        returns (mu, var)
        """

        # ----- 1. hazard-based mixture prior -----
        # with prob (1-h): continue
        # with prob h: reset
        mu_pred = (1 - self.h) * self.mu
        var_pred = (
            (1 - self.h) * (self.var)
            + self.h * self.reset_var
        )

        # ----- 2. Bayesian update (Kalman-style) -----
        K = var_pred / (var_pred + self.obs_var)  # learning rate

        self.mu = mu_pred + K * (observation - mu_pred)
        self.var = (1 - K) * var_pred

        return self.mu, self.var

    def learning_rate(self):
        """Return current effective learning rate"""
        return self.var / (self.var + self.obs_var)
