from math import pi, sqrt
from typing import Any, Union

import numpy as np
import pandas as pd
import torch

from moving_targets.metrics import Metric, DIDI
from moving_targets.util import probabilities


class RegressionWeights(Metric):
    """Computes the weights of the shadow linear regression model."""

    def __init__(self, classification: bool, protected: str, degree: int = 1, name: str = 'w'):
        """
        :param protected:
            The name of the feature to inspect.

        :param classification:
            Whether this is for a classification or a regression task (in the first scenario, binarize the predictions).

        :param degree:
            The degree of the polynomial kernel.

        :param name:
            The name of the metric.
        """
        super(RegressionWeights, self).__init__(name=name)

        self.protected: str = protected
        """The name of the feature to inspect."""

        self.classification: bool = classification
        """Whether this is for a classification or a regression task."""

        self.degree: int = degree
        """The degree of the polynomial kernel."""

    @staticmethod
    def get_weights(x, y, degree: int, use_torch: bool = False) -> Any:
        """
        Computes the the vector <alpha> which is the solution of the following least-square problem:
            argmin || <phi> @ <alpha_tilde> - <psi> ||_2^2
        where <phi> is the zero-centered kernel matrix built from the excluded vector x, and <psi> is the zero-centered
        constant term vector built from the output targets y.

        :param x:
            The vector of features to be excluded.

        :param y:
            The vector of output targets.

        :param degree:
            The kernel degree for the features to be excluded.

        :param use_torch:
            Whether to compute the weights using torch.lstsq or numpy.lstsq

        :return:
            The value of the generalized didi. If return_weights is True, a tuple (<didi>, <alpha_tilde>) is returned.
        """
        phi = [x ** d - (x ** d).mean() for d in np.arange(degree) + 1]
        psi = y - y.mean()
        if use_torch:
            # the 'gelsd' driver allows to have both more precise and more reproducible results
            phi = torch.stack(phi, dim=1)
            alpha, _, _, _ = torch.linalg.lstsq(phi, psi, driver='gelsd')
        else:
            phi = np.stack(phi, axis=1)
            alpha, _, _, _ = np.linalg.lstsq(phi, psi, rcond=None)
        return alpha

    def __call__(self, x, y, p):
        x = x[self.protected].values
        p = probabilities.get_classes(p) if self.classification else p
        alpha = self.get_weights(x=x, y=p, degree=self.degree, use_torch=False)
        return {str(i + 1): a for i, a in enumerate(alpha)}


class BinnedDIDI(DIDI):
    """Performs a binning operation on a continuous protected attribute then computes the DIDI on each bin."""

    def __init__(self,
                 classification: bool,
                 protected: str,
                 bins: int = 2,
                 relative: bool = True,
                 name: str = 'binned_didi'):
        """
        :param classification:
            Whether the DIDI is computed for a classification or a regression task.

        :param protected:
            The name of the protected feature.

        :param bins:
            The number of bins.

        :param relative:
            Whether the DIDI is computed as an absolute or a relative value.

        :param name:
            The name of the metric.
        """
        super(BinnedDIDI, self).__init__(classification, protected, percentage=relative, name=name)

        self.bins: int = bins
        """The number of bins."""

    def __call__(self, x, y, p):
        x = x.copy()
        x[self.protected] = pd.qcut(x[self.protected], q=self.bins).cat.codes
        return super(BinnedDIDI, self).__call__(x, y, p)


class GeneralizedDIDI(Metric):
    def __init__(self,
                 classification: bool,
                 protected: str,
                 degree: int = 1,
                 relative: Union[bool, int] = 1,
                 name: str = 'generalized_didi'):
        """
        :param protected:
            The name of the protected feature.

        :param classification:
            Whether this is for a classification or a regression task (in the first scenario, binarize the predictions).

        :param degree:
            The kernel degree for the excluded feature.

        :param relative:
            If a positive integer k is passed, it computes the relative value with respect to the indicator computed on
            the original targets with kernel k. If True is passed, it assumes k = 1. Otherwise, if False is passed, it
            simply computes the absolute value of the indicator.

        :param name:
            The name of the metric.
        """
        super(GeneralizedDIDI, self).__init__(name=name)

        self.classification: bool = classification
        """Whether we are dealing with a binary classification or a regression task."""

        self.protected: str = protected
        """The feature to be excluded."""

        self.degree: int = degree
        """The kernel degree used for the features to be excluded."""

        self.relative: int = int(relative) if isinstance(relative, bool) else relative
        """The kernel degree to use to compute the metric in relative value, or 0 for absolute value."""

    @staticmethod
    def generalized_didi(x, y, degree: int, use_torch: bool = False, return_weights: bool = False) -> Any:
        """
        Computes the generalized didi as the norm 1 of the vector <alpha_tilde> which is the solution of the following
        least-square problem:
            argmin || <phi> @ <alpha_tilde> - <psi> ||_2^2
        where <phi> is the zero-centered kernel matrix built from the excluded vector x, and <psi> is the zero-centered
        constant term vector built from the output targets y.

        :param x:
            The vector of features to be excluded.

        :param y:
            The vector of output targets.

        :param degree:
            The kernel degree for the features to be excluded.

        :param use_torch:
            Whether to compute the weights using torch.lstsq or numpy.lstsq

        :param return_weights:
            Whether to return the vector <alpha_tilde> along with the didi value or not.

        :return:
            The value of the generalized didi. If return_weights is True, a tuple (<didi>, <alpha_tilde>) is returned.
        """
        alpha_tilde = RegressionWeights.get_weights(x=x, y=y, degree=degree, use_torch=use_torch)
        didi = torch.abs(alpha_tilde).sum() if use_torch else np.abs(alpha_tilde).sum()
        return (didi, alpha_tilde) if return_weights else didi

    def __call__(self, x, y, p):
        x = x[self.protected].values
        p = probabilities.get_classes(p) if self.classification else p
        didi_p = self.generalized_didi(x, p, degree=self.degree)
        didi_y = self.generalized_didi(x, y, degree=self.relative) if self.relative > 0 else 1.0
        return (didi_p / didi_y) if didi_y > 0 else 0.0


class HGR(Metric):
    """Torch-based implementation of the HGR metric obtained from the official repository of "Fairness-Aware Learning
    for Continuous Attributes and Treatments" (https://github.com/criteo-research/continuous-fairness/)."""

    class KDE:
        """A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization. Keep in mind that
        KDE are not scaling well with the number of dimensions and this implementation is not really optimized..."""

        def __init__(self, x_train):
            n, d = x_train.shape
            self.n = n
            self.d = d
            self.bandwidth = (n * (d + 2) / 4.) ** (-1. / (d + 4))
            self.std = self.bandwidth
            self.train_x = x_train

        @staticmethod
        def _unsqueeze_multiple_times(inp, axis, times):
            out = inp
            for i in range(times):
                out = out.unsqueeze(axis)
            return out

        def pdf(self, x):
            s = x.shape
            d = s[-1]
            s = s[:-1]
            assert d == self.d
            data = x.unsqueeze(-2)
            train_x = self._unsqueeze_multiple_times(self.train_x, 0, len(s))
            # noinspection PyTypeChecker
            gaussian_values = torch.exp(-((data - train_x).norm(dim=-1) ** 2 / (self.bandwidth ** 2) / 2))
            pdf_values = gaussian_values.mean(dim=-1) / sqrt(2 * pi) / self.bandwidth
            return pdf_values

    @staticmethod
    def joint_2(x: torch.Tensor, y: torch.Tensor, damping: float = 1e-10, eps: float = 1e-9) -> torch.Tensor:
        # add an eps value to avoid nan vectors in case of very degraded solutions
        x = (x - x.mean()) / (x.std(dim=None) + eps)
        y = (y - y.mean()) / (y.std(dim=None) + eps)
        data = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], -1)
        joint_density = HGR.KDE(data)
        n_bins = int(min(50, 5. / joint_density.std))
        x_centers = torch.linspace(-2.5, 2.5, n_bins)
        y_centers = torch.linspace(-2.5, 2.5, n_bins)
        xx, yy = torch.meshgrid([x_centers, y_centers], indexing='ij')
        grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], -1)
        h2d = joint_density.pdf(grid) + damping
        h2d /= h2d.sum()
        return h2d

    @staticmethod
    def hgr(x: torch.Tensor, y: torch.Tensor, chi2: bool) -> torch.Tensor:
        h2d = HGR.joint_2(x, y)
        marginal_x = h2d.sum(dim=1).unsqueeze(1)
        marginal_y = h2d.sum(dim=0).unsqueeze(0)
        q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
        if chi2:
            return (q ** 2).sum(dim=[0, 1]) - 1.0
        else:
            return torch.linalg.svd(q)[1][1]

    def __init__(self, feature: str, relative: bool = True, chi2: bool = False, name: str = 'hgr'):
        """
        :param feature:
            The name of the feature to inspect.

        :param relative:
            Whether the HGR is computed as an absolute or a relative value.

        :param chi2:
            Whether to return the chi^2 approximation of the HGR or its actual value.

        :param name:
            The name of the metric.
        """
        super(HGR, self).__init__(name=name)

        self.chi2: bool = chi2
        """Whether to return the chi^2 approximation of the HGR or its actual value."""

        self.relative: bool = relative
        """Whether the HGR is computed as an absolute or a relative value."""

        self.feature: str = feature
        """The name of the feature to inspect."""

    def __call__(self, x, y, p):
        x = torch.tensor(x[self.feature].values, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        p = torch.tensor(p, dtype=torch.float)
        hgr_p = HGR.hgr(p, x, chi2=self.chi2).item()
        hgr_y = HGR.hgr(y, x, chi2=self.chi2).item() if self.relative else 1.0
        return hgr_p / hgr_y
