import numpy as np
import random
import mmh3

class xStream:

    '''
    XStream Algorithm
    
    Density-based ensemble outlier detection algorithm.

    Parameters
    ----------
    streamhash
        StreamhashProjection class object.
    deltamax
        List of bin-widths corresponding to half the range of the projected data.
    window_size
        Number of points to observe before replacing the counts in the reference window by those of the current window.
    chains
        Chains class object.
    step
        Counter for the number of points observed.
    cur_window
        Bin-counts for the current window.
    ref_window
        Bin-counts for the reference window.
    '''

    def __init__(
            self,
            num_components=100,
            n_chains=100,
            depth=25,
            window_size=25):
      
        self.streamhash = StreamhashProjection(n_components=num_components,
                                              density=1/3.0,
                                              random_state=42)
        
        deltamax = np.ones(num_components) * 0.5

        deltamax[np.abs(deltamax) <= 0.0001] = 1.0
        self.window_size = window_size

        self.chains = Chains(
            deltamax=deltamax,
            n_chains=n_chains,
            depth=depth)

        self.step = 0
        self.cur_window = []
        self.ref_window = None

    def learn_one(self, X, y=None):
        """Fits the model to next instance.
        """
        self.step += 1

        X = self.streamhash.fit_transform_partial(X)

        X = X.reshape(1, -1)
        self.cur_window.append(X)
        self.chains.fit(X)

        if self.step % self.window_size == 0:
            self.ref_window = self.cur_window
            self.cur_window = []
            deltamax = self._compute_deltamax()
            self.chains.set_deltamax(deltamax)
            self.chains.next_window()

        return self

    def predict_one(self, X):
        """Scores the anomalousness of the next instance.
        """
        X = self.streamhash.fit_transform_partial(X)
        X = X.reshape(1, -1)
        score = self.chains.score(X).flatten()

        return score

    def _compute_deltamax(self):

        mx = np.max(np.concatenate(self.ref_window, axis=0), axis=0)
        mn = np.min(np.concatenate(self.ref_window, axis=0), axis=0)

        deltamax = (mx - mn) / 2.0

        deltamax[np.abs(deltamax) <= 0.0001] = 1.0

        return deltamax
    
    


class StreamhashProjection:
    '''
    Streamhash Projection
    
    Method for subspace-selection and dimensionality reduction via sparse random projections.
    It reduces data dimensionality while accurately preserving distances between points, 
    which facilitates outliers detection.

    Parameters
    ----------
    keys
        Array containing the indexes of the random projections.
    constant
        Constant value used in the hash value computation.
    density
        Fraction of non-zero components in the random projections. Set to 1/3.0 by default.
    n_components
        Number of random projections.
    seed
        Random number seed.  
    '''

    def __init__(self, n_components, density=1/3.0, random_state=None):
        self.keys = np.arange(0, n_components, 1)
        self.constant = np.sqrt(1./density)/np.sqrt(n_components)
        self.density = density
        self.n_components = n_components
        random.seed(random_state)

    def fit_transform_partial(self, X):

        X = X.reshape(1, -1)

        ndim = X.shape[1]

        feature_names = [str(i) for i in range(ndim)]

        R = np.array([[self._hash_string(k, f)
                       for f in feature_names]
                       for k in self.keys])
        

        Y = np.dot(X, R.T).squeeze()

        return Y

    def transform(self, X):
        return self.fit_transform_partial(X)
    
    def _hash_string(self, k, s):
        hash_value = int(mmh3.hash(s, signed=False, seed=k))/(2.0**32-1)
        s = self.density
        if hash_value <= s/2.0:
            return -1 * self.constant
        elif hash_value <= s:
            return self.constant
        else:
            return 0
        

class Chains:
    
    '''
    Ensemble of Chains

    Parameters
    ----------
    n_chains
        Number of chains in the ensemble. Set to 100 by default.
    depth
        Number of feature splits to be performed. Set to 25 by default.
    chains
        Array grouping all the chains.

    '''
    
    def __init__(self, deltamax, n_chains=100, depth=25):

        self.n_chains = n_chains
        self.depth = depth
        self.chains = []

        for i in range(self.n_chains):
          c = Chain(deltamax, depth=self.depth)
          self.chains.append(c)

    def fit(self, X):
        
       for c in self.chains:
            c.fit(X)

    def score(self, X, adjusted=False):
        scores = np.zeros(X.shape[0])
        for c in self.chains:
            scores += c.score(X, adjusted)
        scores /= float(self.n_chains)
        return scores
    
    def next_window(self):
        for c in self.chains:
          c.next_window()
    
    def set_deltamax(self, deltamax):
        for c in self.chains:
            c.deltamax = deltamax
            c.shift = c.rand_arr * deltamax
            

class Chain:

    '''
    Individual Chain

    Method to estimate density at multiple scales 
    The chain approximates the density of a point by counting its nearby neighbors at multiple scales. 
    For every scale or level, a count-min-sketch approximates the bin-counts at that level.
    Non-stationarity of data is handled by maintaining separate bin-counts for an alternating pair of windows
    containing ψ points each, termed as current and reference windows. 
    
    Parameters
    ----------
    k
        Number of components or random projections.
    deltamax
        List of bin-widths corresponding to half the range of the projected data.
    depth
        Number of feature splits to be performed. Set to 25 by default.
    fs
        List containing the randomly selected split features or dimensions.
    cmsketches_ref
        Reference count-min-sketches corresponding to the reference window.
    cmsketches_cur
        Current count-min-sketches corresponding to the current window.
    rand_arr
        List of uniform random numbers used to compute the shift values.
    shift
        List containing the uniform shift value for every component.
    is_first_window
        Boolean value indicating whether the window under consideration is the first one or not.
    '''

    def __init__(self, deltamax, depth=25):
        k = len(deltamax)
        self.deltamax = deltamax # feature ranges
        self.depth = depth
        self.fs = [np.random.randint(0, k) for d in range(depth)]
        self.cmsketches_ref = [{} for i in range(depth)] * depth
        self.cmsketches_cur = [{} for i in range(depth)] * depth
        self.rand_arr = np.random.rand(k)
        self.shift = self.rand_arr * deltamax
        self.is_first_window = True

    def fit(self, X, verbose=False):#, update=False):
        prebins = np.zeros(X.shape, dtype=float)
        depthcount = np.zeros(len(self.deltamax), dtype=int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:,f] = (X[:,f] + self.shift[f])/self.deltamax[f]
            else:
                prebins[:,f] = 2.0*prebins[:,f] - self.shift[f]/self.deltamax[f]

            if self.is_first_window:
              cmsketch = self.cmsketches_ref[depth] 
              for prebin in prebins:
                  l = tuple(np.floor(prebin).astype(int))
                  if not l in cmsketch:
                      cmsketch[l] = 0
                  cmsketch[l] += 1

              self.cmsketches_ref[depth] = cmsketch
              self.cmsketches_cur[depth] = cmsketch

            else:
              cmsketch = self.cmsketches_cur[depth] 
              for prebin in prebins:
                  l = tuple(np.floor(prebin).astype(int))
                  if not l in cmsketch:
                      cmsketch[l] = 0
                  cmsketch[l] += 1
              
              self.cmsketches_cur[depth] = cmsketch

        return self

    def bincount(self, X):
        scores = np.zeros((X.shape[0], self.depth))
        prebins = np.zeros(X.shape, dtype=float)
        depthcount = np.zeros(len(self.deltamax), dtype=int)
        for depth in range(self.depth):
            f = self.fs[depth] 
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:,f] = (X[:,f] + self.shift[f])/self.deltamax[f]
            else:
                prebins[:,f] = 2.0*prebins[:,f] - self.shift[f]/self.deltamax[f]

            cmsketch = self.cmsketches_ref[depth]
            for i, prebin in enumerate(prebins):
                l = tuple(np.floor(prebin).astype(int))
                if not l in cmsketch:
                    scores[i,depth] = 0.0
                else:
                    scores[i,depth] = cmsketch[l]

        return scores

    def score(self, X, adjusted=False):
        # scale score logarithmically to avoid overflow:
        #    score = min_d [ log2(bincount x 2^d) = log2(bincount) + d ]
        scores = self.bincount(X)
        depths = np.array([d for d in range(1, self.depth+1)])
        scores = np.log2(1.0 + scores) + depths # add 1 to avoid log(0)
        return np.min(scores, axis=1)

    def next_window(self):
        self.is_first_window = False
        self.cmsketches_ref = self.cmsketches_cur
        self.cmsketches_cur = [{} for _ in range(self.depth)] * self.depth