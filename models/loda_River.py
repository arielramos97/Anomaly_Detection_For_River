import numpy as np

class LODA():
    """
    Lightweight Online Detector of Anomalies
    
    Outlier detection algorithm that computes the likelihood of 
    data points using an ensemble of one-dimensional histograms.
    
    Parameters
    ----------
    num_bins
        Number of bins of the histogram.
    num_random_cuts
        Number of random cut projections.
    """

    def __init__(self, num_bins=10, num_random_cuts=100):
        self.to_init = True
        self.n_bins = num_bins
        self.n_random_cuts = num_random_cuts

    def learn_one(self, X, y=None):
        """Fits the model to next instance.
        """
        if self.to_init:
            self.num_features = X.shape[0]
            self.weights = np.ones(
                self.n_random_cuts,
                dtype=np.float) / self.n_random_cuts
            self.projections_ = np.random.randn(
                self.n_random_cuts, self.num_features)
            self.histograms_ = np.zeros((self.n_random_cuts, self.n_bins))
            self.limits_ = np.zeros((self.n_random_cuts, self.n_bins + 1))

            n_nonzero_components = np.sqrt(self.num_features)
            self.n_zero_components = self.num_features - \
                np.int(n_nonzero_components)

            self.to_init = False

        X = X.reshape(1, -1)

        for i in range(self.n_random_cuts):
            rands = np.random.permutation(self.num_features)[
                :self.n_zero_components]
            self.projections_[i, rands] = 0.
            projected_data = self.projections_[i, :].dot(X.T)
            self.histograms_[i, :], self.limits_[i, :] = np.histogram(
                projected_data, bins=self.n_bins, density=False)
            self.histograms_[i, :] += 1e-12
            self.histograms_[i, :] /= np.sum(self.histograms_[i, :])

        return self

    def predict_one(self, X):
        """Scores the anomalousness of the next instance.
        """
        X = X.reshape(1, -1)

        pred_scores = np.zeros([X.shape[0], 1])
        for i in range(self.n_random_cuts):
            projected_data = self.projections_[i, :].dot(X.T)
            inds = np.searchsorted(self.limits_[i, :self.n_bins - 1],
                                   projected_data, side='left')
            pred_scores[:, 0] += -self.weights[i] * np.log(
                self.histograms_[i, inds])
        pred_scores /= self.n_random_cuts

        return pred_scores.ravel()