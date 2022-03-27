from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet

pdfLambda = lambda mu,var,x: (1.0 / (var * ((2*np.pi)**0.5))) * np.exp(-0.5*((x - mu) / float(var)) ** 2)

def multivarPdf(X, mu, cov):
    size = len(X)
    if size == len(mu) and (size, size) == cov.shape:
        deter = det(cov)
        if deter == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0 / ((2 * np.pi)**(float(size) / 2) * (deter**(1.0 / 2)))
        x_mu = np.matrix(X - mu)
        invar = inv(cov)
        result = np.exp(-0.5 * (x_mu * invar * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X)
        if self.biased_:
            self.var_ = np.mean(abs(X-self.mu_))
        else:
            self.var_ = X.var(ddof=1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        # denom = (2*np.pi*self.var_)**0.5
        # num = np.exp(-(X-self.mu_)**2/(2*self.var_))
        # pdf = np.exp(-1*(((X-self.mu_)**2)/(2*self.var_)))/((2*np.pi*self.var_)**0.5)
        return np.array(np.array([pdfLambda(self.mu_, self.var_, v) for v in X]))

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        likelihoodSum = 0
        constant = (-1.0) / (2*sigma)
        for xi in X:
            likelihoodSum += pow(xi-mu, 2)
        # denom = (2*np.pi*var)**0.5
        # num = np.exp(-(X-mu)**2/(2*var))
        # pdf = num/denom
        # logLikelihood = ((1/(2*np.pi*sigma**2)**0.5)**len(X))*np.exp((-1*(1/2*sigma**2)*np.sum((X-mu)**2)))
        # #return sum([np.log(x) for x in pdf])

        return (- len(X) / 2) * np.log(2*np.pi) + np.log(sigma) + constant * likelihoodSum


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, 0)
        self.cov_ = np.cov(X.T)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        return np.array([multivarPdf(xi, self.mu_, self.cov_) for xi in X])

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        residuals = X-mu

        calcLogLikelihood = lambda resid, invariant, logDet, dim: -0.5 * (logDet + (resid.T @ invariant @ resid) + (dim * np.log(2 * np.pi)))
        invariantMat = inv(cov)
        logDeter = slogdet(cov)
        d = cov.shape[0]
        logPDF = np.apply_along_axis(lambda xi: calcLogLikelihood(xi, invariantMat, logDeter, d), 1, residuals)
        return logPDF.sum()


if __name__ == "__main__":
    quizX = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    univar = UnivariateGaussian().fit(quizX)
    print(UnivariateGaussian.log_likelihood(10,1,quizX))
    pass