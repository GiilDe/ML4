import scipy
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt


class Distribution(object):

    def __init__(self):
        self.dist_names = ['norm', 'lognorm', 'expon', 'alpha', 'beta', 'arcsine', 'weibull_min', 'weibull_max', 'rayleigh']
        self.dist_results = []
        self.params = {}

        self.DistributionName = ""
        self.PValue = 0
        self.Param = None

        self.isFitted = False

    def Fit(self, y):
        self.dist_results = []
        self.params = {}
        for dist_name in self.dist_names:
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(y)

            self.params[dist_name] = param
            # Applying the Kolmogorov-Smirnov test
            D, p = scipy.stats.kstest(y, dist_name, args=param)
            self.dist_results.append((dist_name, p))

        # select the best fitted distribution
        sel_dist, p = (max(self.dist_results, key=lambda item: item[1]))
        # store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.PValue = p

        self.isFitted = True
        return self.DistributionName, self.PValue

    def Random(self, n=1):
        if self.isFitted:
            dist_name = self.DistributionName
            param = self.params[dist_name]
            # initiate the scipy distribution
            dist = getattr(scipy.stats, dist_name)
            return dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=n)
        else:
            raise ValueError('Must first run the Fit method.')

    def pdf(self, x):
        if self.isFitted:
            dist_name = self.DistributionName
            param = self.params[dist_name]
            # initiate the scipy distribution
            dist = getattr(scipy.stats, dist_name)
            return dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
        else:
            raise ValueError('Must first run the Fit method.')