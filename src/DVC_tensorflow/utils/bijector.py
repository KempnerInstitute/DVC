import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class NormalCDF(tfb.Bijector):
    """Bijector that encodes normal CDF and inverse CDF functions.

    We follow the convention that the `inverse` represents the CDF
    and `forward` the inverse CDF (the reason for this convention is
    that inverse CDF methods for sampling are expressed a little more
    tersely this way).

    """

    def __init__(self, loc, scale):
        self.normal_dist = tfd.Normal(loc=loc, scale=scale)
        super(NormalCDF, self).__init__(
            forward_min_event_ndims=0,
            validate_args=False,
            name="NormalCDF")

    def forward(self, y):
        # Inverse CDF of normal distribution.
        return self.normal_dist.quantile(y)

    def inverse(self, x):
        # CDF of normal distribution.
        return self.normal_dist.cdf(x)

    def inverse_log_det_jacobian(self, x):
        # Log PDF of the normal distribution.
        return self.normal_dist.log_prob(x)
    

class GammaCDF(tfb.Bijector):
    """Bijector that encodes normal CDF and inverse CDF functions.

    We follow the convention that the `inverse` represents the CDF
    and `forward` the inverse CDF (the reason for this convention is
    that inverse CDF methods for sampling are expressed a little more
    tersely this way).

    """

    def __init__(self,concentration,rate):
        self.gamma_dist = tfd.Gamma(concentration=concentration, rate=rate)
        super(GammaCDF, self).__init__(
            forward_min_event_ndims=0,
            validate_args=False,
            name="GammaCDF")

    def forward(self, y):
        # Inverse CDF of normal distribution.
        return self.gamma_dist.quantile(y)

    def inverse(self, x):
        # CDF of normal distribution.
        return self.gamma_dist.cdf(x)

    def inverse_log_det_jacobian(self, x):
        # Log PDF of the normal distribution.
        return self.gamma_dist.log_prob(x)