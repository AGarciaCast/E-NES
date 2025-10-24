from flax import linen as nn


class BaseThreewayInvariants(nn.Module):

    def setup(self):
        self.symmetric = False
