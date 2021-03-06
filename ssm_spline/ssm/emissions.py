import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd.scipy.linalg import block_diag

from ssm.util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_slds_args_not_none, logistic, logit, softplus, inv_softplus, spline_func, logistic_func, mnn_func, mnn_func2, mnn_func3
from ssm.preprocessing import interpolate_data, pca_with_imputation


# Observation models for SLDS
class _Emissions(object):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        self.N, self.K, self.D, self.M, self.single_subspace = \
            N, K, D, M, single_subspace

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        pass

    def initialize_from_arhmm(self, arhmm, pca):
        pass

    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag, x):
        raise NotImplementedError

    def forward(self, x, input=None, tag=None):
        raise NotImplemented

    @ensure_args_not_none
    def invert(self, data, input=None, mask=None, tag=None):
        raise NotImplemented

    def sample(self, z, x, input=None, tag=None):
        raise NotImplementedError

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError


# Many emissions models start with a linear layer
class _LinearEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_LinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize linear layer
        # Use the rational Cayley transform to parameterize an orthogonal emission matrix
        assert N > D
        self._Ms = npr.randn(1, D, D) if single_subspace else npr.randn(K, D, D)
        self._As = npr.randn(1, N-D, D) if single_subspace else npr.randn(K, N-D, D)
        self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)

        # Set the emission matrix to be a random orthogonal matrix
        C0 = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        for k in range(C0.shape[0]):
            C0[k] = np.linalg.svd(C0[k], full_matrices=False)[0]
        self.Cs = C0

    @property
    def Cs(self):
        # See https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.5b02015
        # for a derivation of the rational Cayley transform.
        D = self.D
        T = lambda X: np.swapaxes(X, -1, -2)

        Bs = 0.5 * (self._Ms - T(self._Ms))    # Bs is skew symmetric
        Fs = np.matmul(T(self._As), self._As) - Bs
        trm1 = np.concatenate((np.eye(D) - Fs, 2 * self._As), axis=1)
        trm2 = np.eye(D) + Fs
        Cs = T(np.linalg.solve(T(trm2), T(trm1)))
        assert np.allclose(
            np.matmul(T(Cs), Cs),
            np.tile(np.eye(D)[None, :, :], (Cs.shape[0], 1, 1))
            )
        return Cs

    @Cs.setter
    def Cs(self, value):
        N, D = self.N, self.D
        T = lambda X: np.swapaxes(X, -1, -2)

        # Make sure value is the right shape and orthogonal
        Keff = 1 if self.single_subspace else self.K
        assert value.shape == (Keff, N, D)
        assert np.allclose(
            np.matmul(T(value), value),
            np.tile(np.eye(D)[None, :, :], (Keff, 1, 1))
            )

        Q1s, Q2s = value[:, :D, :], value[:, D:, :]
        Fs = T(np.linalg.solve(T(np.eye(D) + Q1s), T(np.eye(D) - Q1s)))
        # Bs = 0.5 * (T(Fs) - Fs) = 0.5 * (self._Ms - T(self._Ms)) -> _Ms = T(Fs)
        self._Ms = T(Fs)
        self._As = 0.5 * np.matmul(Q2s, np.eye(D) + Fs)
        assert np.allclose(self.Cs, value)

    @property
    def params(self):
        return self._As, self._Ms, self.Fs, self.ds

    @params.setter
    def params(self, value):
        self._As, self._Ms, self.Fs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self._As = self._As[perm]
            self._Ms = self._Ms[perm]
            self.Fs = self.Fs[perm]
            self.ds = self.ds[perm]

    def _invert(self, data, input, mask, tag):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        assert self.single_subspace, "Can only invert with a single emission model"

        T = data.shape[0]
        C, F, d = self.Cs[0], self.Fs[0], self.ds[0]
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        bias = input.dot(F.T) + d

        if not np.all(mask):
            data = interpolate_data(data, mask)
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            for itr in range(25):
                q_mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (q_mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)

    def forward(self, x, input, tag):
        # print(np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0].shape)

        return np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0] \
             + np.matmul(self.Fs[None, ...], input[:, None, :, None])[:, :, :, 0] \
             + self.ds

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20, smooth=0):
        Keff = 1 if self.single_subspace else self.K

        if smooth > 0:
            # TODO: smooth the data, if requested, with a Gaussian filter
            pass

        # N_smooth=3
        # datas = [np.convolve(data, np.ones((N_smooth,))/N_smooth, mode='same') for data in datas]


        # First solve a linear regression for data given input
        if self.M > 0:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.vstack(inputs), np.vstack(datas))
            self.Fs = np.tile(lr.coef_[None, :, :], (Keff, 1, 1))

        # Compute residual after accounting for input
        resids = [data - np.dot(input, self.Fs[0].T) for data, input in zip(datas, inputs)]

        # Run PCA to get a linear embedding of the data
        pca, xs = pca_with_imputation(self.D, resids, masks, num_iters=num_iters)

        self.Cs = np.tile(pca.components_.T[None, :, :], (Keff, 1, 1))
        self.ds = np.tile(pca.mean_[None, :], (Keff, 1))

        return pca


class _CompoundLinearEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True,
                 N_vec=None, D_vec=None, **kwargs):
        """
        N_vec, D_vec are the sizes of the constituent emission models.
        Assume N_vec and D_vec are lists/tuples/arrays of length G and

        N_vec = [N_1, ..., N_P] indicates that the first group of neurons
        is size N_1, the P-th populations is size N_P.  Likewise for D_vec.
        We will assume that the data is grouped in the same way.

        We require sum(N_vec) == N and sum(D_vec) == D.
        """
        super(_CompoundLinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        assert isinstance(N_vec, (np.ndarray, list, tuple))
        N_vec = np.array(N_vec, dtype=int)
        assert np.sum(N_vec) == N

        assert isinstance(D_vec, (np.ndarray, list, tuple)) and len(D_vec) == len(N_vec)
        D_vec = np.array(D_vec, dtype=int)
        assert np.sum(D_vec) == D

        self.N_vec, self.D_vec = N_vec, D_vec

        # Save the number of subpopulations
        self.P = len(self.N_vec)

        # The main purpose of this class is to wrap a bunch of emissions instances
        self.emissions_models = [_LinearEmissions(n, K, d) for n, d in zip(N_vec, D_vec)]

    @property
    def Cs(self):
        if self.single_subspace:
            return np.array([block_diag(*[em.Cs[0] for em in self.emissions_models])])
        else:
            return np.array([block_diag(*[em.Cs[k] for em in self.emissions_models])
                             for k in range(self.K)])

    @property
    def ds(self):
        return np.concatenate([em.ds for em in self.emissions_models], axis=1)

    @property
    def Fs(self):
        return np.concatenate([em.Fs for em in self.emissions_models], axis=1)

    @property
    def params(self):
        return [em.params for em in self.emissions_models]

    @params.setter
    def params(self, value):
        assert len(value) == self.P
        for em, v in zip(self.emissions_models, value):
            em.params = v

    def permute(self, perm):
        for em in self.emissions_models:
            em.permute(perm)

    def _invert(self, data, input, mask, tag):
        assert data.shape[1] == self.N
        N_offsets = np.cumsum(self.N_vec)[:-1]
        states = []
        for em, dp, mp in zip(self.emissions_models,
                            np.split(data, N_offsets, axis=1),
                            np.split(mask, N_offsets, axis=1)):
            states.append(em._invert(dp, input, mp, tag))
        return np.column_stack(states)

    def forward(self, x, input, tag):
        assert x.shape[1] == self.D
        D_offsets = np.cumsum(self.D_vec)[:-1]
        datas = []
        for em, xp in zip(self.emissions_models, np.split(x, D_offsets, axis=1)):
            datas.append(em.forward(xp, input, tag))
        return np.concatenate(datas, axis=2)

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        for data in datas:
            assert data.shape[1] == self.N

        N_offsets = np.cumsum(self.N_vec)[:-1]
        pcas = []

        split_datas = list(zip(*[np.split(data, N_offsets, axis=1) for data in datas]))
        split_masks = list(zip(*[np.split(mask, N_offsets, axis=1) for mask in masks]))
        assert len(split_masks) == len(split_datas) == self.P

        for em, dps, mps in zip(self.emissions_models, split_datas, split_masks):
            pcas.append(em._initialize_with_pca(dps, inputs, mps, tags))

        # Combine the PCA objects
        from sklearn.decomposition import PCA
        pca = PCA(self.D)
        pca.components_ = block_diag(*[p.components_ for p in pcas])
        pca.mean_ = np.concatenate([p.mean_ for p in pcas])
        # Not super pleased with this, but it should work...
        pca.noise_variance_ = np.concatenate([p.noise_variance_ * np.ones(n)
                                              for p,n in zip(pcas, self.N_vec)])
        return pca

# Linear emissions layer w/o the orthogonality constraint
class _LinearNonOrthEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_LinearNonOrthEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize linear layer
        # Use the rational Cayley transform to parameterize an orthogonal emission matrix
        assert N > D
        self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)

        # Set the emission matrix to be a random orthogonal matrix
        C0 = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        for k in range(C0.shape[0]):
            C0[k] = np.linalg.svd(C0[k], full_matrices=False)[0]
        self.Cs = C0

    @property
    def params(self):
        return self.Cs, self.Fs, self.ds

    @params.setter
    def params(self, value):
        self.Cs, self.Fs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self.Cs = self.Cs[perm]
            self.Fs = self.Fs[perm]
            self.ds = self.ds[perm]

    def _invert(self, data, input, mask, tag):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        assert self.single_subspace, "Can only invert with a single emission model"

        T = data.shape[0]
        C, F, d = self.Cs[0], self.Fs[0], self.ds[0]
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        bias = input.dot(F.T) + d

        if not np.all(mask):
            data = interpolate_data(data, mask)
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            for itr in range(25):
                q_mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (q_mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)

    def forward(self, x, input, tag):
        return np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0] \
             + np.matmul(self.Fs[None, ...], input[:, None, :, None])[:, :, :, 0] \
             + self.ds

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20, smooth=0):
        Keff = 1 if self.single_subspace else self.K

        if smooth > 0:
            # TODO: smooth the data, if requested, with a Gaussian filter
            pass

        # First solve a linear regression for data given input
        if self.M > 0:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.vstack(inputs), np.vstack(datas))
            self.Fs = np.tile(lr.coef_[None, :, :], (Keff, 1, 1))

        # Compute residual after accounting for input
        resids = [data - np.dot(input, self.Fs[0].T) for data, input in zip(datas, inputs)]

        # Run PCA to get a linear embedding of the data
        pca, xs = pca_with_imputation(self.D, resids, masks, num_iters=num_iters)

        self.Cs = np.tile(pca.components_.T[None, :, :], (Keff, 1, 1))
        self.ds = np.tile(pca.mean_[None, :], (Keff, 1))

        return pca

class _CompoundLinearNonOrthEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True,
                 N_vec=None, D_vec=None, **kwargs):
        """
        N_vec, D_vec are the sizes of the constituent emission models.
        Assume N_vec and D_vec are lists/tuples/arrays of length G and

        N_vec = [N_1, ..., N_P] indicates that the first group of neurons
        is size N_1, the P-th populations is size N_P.  Likewise for D_vec.
        We will assume that the data is grouped in the same way.

        We require sum(N_vec) == N and sum(D_vec) == D.
        """
        super(_CompoundLinearNonOrthEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        assert isinstance(N_vec, (np.ndarray, list, tuple))
        N_vec = np.array(N_vec, dtype=int)
        assert np.sum(N_vec) == N

        assert isinstance(D_vec, (np.ndarray, list, tuple)) and len(D_vec) == len(N_vec)
        D_vec = np.array(D_vec, dtype=int)
        assert np.sum(D_vec) == D

        self.N_vec, self.D_vec = N_vec, D_vec

        # Save the number of subpopulations
        self.P = len(self.N_vec)

        # The main purpose of this class is to wrap a bunch of emissions instances
        self.emissions_models = [_LinearNonOrthEmissions(n, K, d) for n, d in zip(N_vec, D_vec)]

    @property
    def Cs(self):
        if self.single_subspace:
            return np.array([block_diag(*[em.Cs[0] for em in self.emissions_models])])
        else:
            return np.array([block_diag(*[em.Cs[k] for em in self.emissions_models])
                             for k in range(self.K)])

    @property
    def ds(self):
        return np.concatenate([em.ds for em in self.emissions_models], axis=1)

    @property
    def Fs(self):
        return np.concatenate([em.Fs for em in self.emissions_models], axis=1)

    @property
    def params(self):
        return [em.params for em in self.emissions_models]

    @params.setter
    def params(self, value):
        assert len(value) == self.P
        for em, v in zip(self.emissions_models, value):
            em.params = v

    def permute(self, perm):
        for em in self.emissions_models:
            em.permute(perm)

    def _invert(self, data, input, mask, tag):
        assert data.shape[1] == self.N
        N_offsets = np.cumsum(self.N_vec)[:-1]
        states = []
        for em, dp, mp in zip(self.emissions_models,
                            np.split(data, N_offsets, axis=1),
                            np.split(mask, N_offsets, axis=1)):
            states.append(em._invert(dp, input, mp, tag))
        return np.column_stack(states)

    def forward(self, x, input, tag):
        assert x.shape[1] == self.D
        D_offsets = np.cumsum(self.D_vec)[:-1]
        datas = []
        for em, xp in zip(self.emissions_models, np.split(x, D_offsets, axis=1)):
            datas.append(em.forward(xp, input, tag))
        return np.concatenate(datas, axis=2)

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        for data in datas:
            assert data.shape[1] == self.N

        N_offsets = np.cumsum(self.N_vec)[:-1]
        pcas = []

        split_datas = list(zip(*[np.split(data, N_offsets, axis=1) for data in datas]))
        split_masks = list(zip(*[np.split(mask, N_offsets, axis=1) for mask in masks]))
        assert len(split_masks) == len(split_datas) == self.P

        for em, dps, mps in zip(self.emissions_models, split_datas, split_masks):
            pcas.append(em._initialize_with_pca(dps, inputs, mps, tags))

        # Combine the PCA objects
        from sklearn.decomposition import PCA
        pca = PCA(self.D)
        pca.components_ = block_diag(*[p.components_ for p in pcas])
        pca.mean_ = np.concatenate([p.mean_ for p in pcas])
        # Not super pleased with this, but it should work...
        pca.noise_variance_ = np.concatenate([p.noise_variance_ * np.ones(n)
                                              for p,n in zip(pcas, self.N_vec)])
        return pca



# Sometimes we just want a bit of additive noise on the observations
class _IdentityEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_IdentityEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)
        assert N == D

    def forward(self, x, input):
        return x

    def _invert(self, data, input, mask, tag):
        """
        Inverse is just the data
        """
        return np.copy(data)


# Spline Emissions
class _SplineEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True,comb_type='add',switch=False,switch_iter=1000,n_kn=13, init_params=None, **kwargs):
        assert single_subspace, "_NeuralNetworkEmissions only supports `single_subspace=True`"
        super(_SplineEmissions, self).__init__(N, K, D, M=M, single_subspace=True)

        self.comb_type=comb_type
        # Initialize the parameters
        # n_kn=16
        self.n_kn=n_kn
        # self.ps=npr.randn(N,n_kn,D)
        if init_params is None:
            self.ps=-6*np.ones([N,n_kn,D])
        else:
            self.ps=init_params
        # self.ps[:,4:(n_kn-4),:]=-1.5
        # self.ps[6]=.38
        # self.ps[:,1:,:]=-2.25
        # print(self.ps.shape)
        # self.ps[0]=-10
        self.iter_count=0
        self.switch=switch
        self.switch_iter=switch_iter

    @property
    def params(self):
        return self.ps

    @params.setter
    def params(self, value):
        self.ps = value

    def permute(self, perm):
        pass

    def forward(self, x, input, tag):
        # # output = [spline_func(x[:,None,0,None],p_i) for p_i in self.ps]
        # # output = [spline_func(x,p_i) for p_i in self.ps] #good before
        # output = [spline_func(x,p_i) for p_i in self.ps]
        # # output = [spline_func(x[:,None,i,None],p_i) for (p_i, i) in zip(self.ps, range(N)]
        # # print(len(output))
        # # print(output[0].shape)
        # # print(x.shape)
        # # print(np.hstack(output).shape)
        # # print(np.vstack(output).shape)
        # temp=np.hstack(output)
        # temp2=np.expand_dims(temp,1)
        # # print(temp2.shape)
        # # return np.vstack(output)
        # return temp2
        if self.comb_type=='mult':
            final_output=1
            for d in range(self.D):
                # print("d",d)
                # print("p shape",self.ps.shape)
                # temp_p=self.ps[:,:,d:d+1]
                output = [spline_func(x[:,d:d+1],p_i[:,d:d+1],self.n_kn) for p_i in self.ps]
                temp=np.hstack(output)
                temp2=np.expand_dims(temp,1)
                final_output=final_output*temp2
            return final_output
        elif self.comb_type=='add':
            final_output=0
            for d in range(self.D):
                # print("d",d)
                # print("p shape",self.ps.shape)
                # temp_p=self.ps[:,:,d:d+1]
                output = [spline_func(x[:,d:d+1],p_i[:,d:d+1],self.n_kn) for p_i in self.ps]
                temp=np.hstack(output)
                temp2=np.expand_dims(temp,1)
                final_output=final_output+temp2
            return final_output
        else:
            NotImplementedError('Incorrect type of latent combination')



    def _invert(self, data, input, mask, tag):
        """
        Inverse is... who knows!
        """
        # return npr.randn(data.shape[0], self.D)
        # m=np.mean(data,axis=1).reshape(-1,1)
        # return 1*m/np.max(m)

        # temp=npr.rand(data.shape[0], self.D)-0.5
        # m=np.mean(data,axis=1).reshape(-1,1)
        # temp[:,0:1]=m/np.max(m)-0.5
        if tag is None:
            temp=1.6*npr.rand(data.shape[0], self.D)-0.8
            m=np.mean(data,axis=1).reshape(-1,1)
            # temp[:,0:1]=1.6*m/np.max(m)-0.8+.01*npr.rand(data.shape[0], 1)
            if np.all(m==0):
                temp[:,0:1]=-0.8
            else:
                temp[:,0:1]=1.6*m/np.max(m)-0.8
        else:
            temp=tag

        return temp

    def log_prior(self):
        self.iter_count=self.iter_count+1
        beta=1
        beta2=0
        # print(self.iter_count)
        if self.switch:
            if self.iter_count>self.switch_iter:
                delta_ys=softplus(self.ps)
                smooth_spline_pens=[np.sum(np.abs(np.diff(delta_ys[i],axis=0))) for i in range(len(self.ps))]
                return -beta*np.sum(delta_ys**2) - beta2 * np.sum(smooth_spline_pens)
            else:
                return 0
        else:
            delta_ys=softplus(self.ps)
            # smooth_spline_pens=[np.sum(np.abs(np.diff(delta_ys[i],axis=0))) for i in range(len(self.ps))]
            smooth_spline_pens=[np.sum(np.abs(np.diff(np.diff(delta_ys[i],axis=0),axis=0))) for i in range(len(self.ps))]
            return -beta*np.sum(delta_ys**2) - beta2 * np.sum(smooth_spline_pens)
            # return 0



# Spline Emissions
class _LogisticEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True,comb_type='add',init_params=None, **kwargs):
        assert single_subspace, "_NeuralNetworkEmissions only supports `single_subspace=True`"
        super(_LogisticEmissions, self).__init__(N, K, D, M=M, single_subspace=True)

        self.comb_type=comb_type
        # Initialize the parameters
        # n_kn=16
        if init_params is None:
            self.ps=np.zeros([N,3,D])
            for i in range(N):
                for j in range(D):
                    self.ps[i,:,j]=np.array([10.,0.,.5])
                    # self.ps[i,:,j]=np.array([10.,0.,1.])
        else:
            # self.ps=np.array(init_params)
            self.ps=init_params


    @property
    def params(self):
        return self.ps

    @params.setter
    def params(self, value):
        self.ps = value

    def permute(self, perm):
        pass

    def forward(self, x, input, tag):
        # # output = [spline_func(x[:,None,0,None],p_i) for p_i in self.ps]
        # # output = [spline_func(x,p_i) for p_i in self.ps] #good before
        # output = [spline_func(x,p_i) for p_i in self.ps]
        # # output = [spline_func(x[:,None,i,None],p_i) for (p_i, i) in zip(self.ps, range(N)]
        # # print(len(output))
        # # print(output[0].shape)
        # # print(x.shape)
        # # print(np.hstack(output).shape)
        # # print(np.vstack(output).shape)
        # temp=np.hstack(output)
        # temp2=np.expand_dims(temp,1)
        # # print(temp2.shape)
        # # return np.vstack(output)
        # return temp2
        if self.comb_type=='mult':
            final_output=1
            for d in range(self.D):
                output = [logistic_func(x[:,d:d+1],p_i[:,d:d+1]) for p_i in self.ps]
                temp=np.hstack(output)
                temp2=np.expand_dims(temp,1)
                final_output=final_output*temp2
            return final_output
        elif self.comb_type=='add':
            final_output=0
            for d in range(self.D):
                # print("d",d)
                # print("p shape",self.ps.shape)
                # temp_p=self.ps[:,:,d:d+1]
                output = [logistic_func(x[:,d:d+1],p_i[:,d:d+1]) for p_i in self.ps]
                temp=np.hstack(output)
                temp2=np.expand_dims(temp,1)
                final_output=final_output+temp2
            return final_output
        else:
            NotImplementedError('Incorrect type of latent combination')



    def _invert(self, data, input, mask, tag):
        """
        Inverse is... who knows!
        """
        # return npr.randn(data.shape[0], self.D)
        # m=np.mean(data,axis=1).reshape(-1,1)
        # return 1*m/np.max(m)

        # temp=npr.rand(data.shape[0], self.D)-0.5
        # m=np.mean(data,axis=1).reshape(-1,1)
        # temp[:,0:1]=m/np.max(m)-0.5

        temp=1.6*npr.rand(data.shape[0], self.D)-0.8
        # m=np.mean(data,axis=1).reshape(-1,1)

        # #Initialize first latent w/ mean
        if tag is None:
            print("no tag")
            m=np.mean(data*mask,axis=1).reshape(-1,1)
            if np.all(m==0):
                temp[:,0:1]=-0.8
            else:
                temp[:,0:1]=1.6*m/np.max(m)-0.8#+.01*npr.rand(data.shape[0], 1)

        else:
            print("tag")
            temp=tag

        #Initialize latents w/ means of subsets of neurons
        # for d in range(self.D):
        #     m=np.mean(data[:,int(data.shape[1]*d/self.D):int(data.shape[1]*(d+1)/self.D)])
        #     if np.all(m==0):
        #         temp[:,d]=-0.8
        #     else:
        #         temp[:,d]=1.6*m/np.max(m)-0.8


        return temp

    def log_prior(self):
            gamma=0 #was 10 #or 100
            vals=[self.ps[i][0,:]**2 for i in range(len(self.ps))]
            # print(vals)
            return -gamma*np.sum(vals)#np.sum(self.ps[:,1,:]**2)

            # return 0


class _MonotonicNeuralNetworkEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, hidden_layer_sizes=(50,),init_params=None, single_subspace=True):
        assert single_subspace, "_NeuralNetworkEmissions only supports `single_subspace=True`"
        super(_MonotonicNeuralNetworkEmissions, self).__init__(N, K, D, M=M, single_subspace=True)

        # Initialize the neural network weights
        assert N > D
        layer_sizes = (D + M,) + hidden_layer_sizes + (1,)

        self.iter_count=0

        # print("w_orig",[npr.randn(m, n) * np.sqrt(1. / m) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])])
        # print("b_orig",[npr.randn(n) for n in layer_sizes[1:]])


        # print("w_new",[[npr.randn(m, n) * np.sqrt(1. / m) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])] for i in range(2)])
        # self.N=N

        # weights=[]
        if init_params is None:
            biases=[]
            for i in range(N):
                # weights.append([npr.randn(m, n) * np.sqrt(1. / m) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])])
                biases.append([npr.randn(n) for n in layer_sizes[1:]])
            self.weights = [[npr.randn(m, n) * np.sqrt(1. / m) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])] for i in range(N)]
            # self.biases = [[npr.randn(n) for n in layer_sizes[1:]] for i in range(N)]
            # self.weights=weights
            self.biases=biases
        else:
            self.weights, self.biases = init_params


        # print("w_new",self.weights[0])
        # print("b_new",self.biases[0])
        # print("w_new",self.weights)
        # print("b_new",self.biases)
        #
        # print("w_tmp",self.weights[0])
        # i=0
        # for ws,bs in zip(self.weights[i],self.biases[i]):
        #     print("w",ws)
        #     print("b",bs)

    @property
    def params(self):
        return self.weights, self.biases

    @params.setter
    def params(self, value):
        self.weights, self.biases = value

    def permute(self, perm):
        pass

    def forward(self, x, input, tag):
        inputs = np.column_stack((x, input))

        output=[]

        for i in range(len(self.weights)):
        #     for ws,bs in zip(self.weights[i],self.biases[i]):
        #         print("ws",ws)
        #         print("bs",bs)
        #         print("next")
        #         for w,b in zip(ws,bs):
        #             print("w",w)
        #             print("b",b)

            # output.append(mnn_func(x,ws,bs) for ws,bs in zip(self.weights[i],self.biases[i]))
            if self.iter_count<-1:
                output.append(mnn_func(x,self.weights[i],self.biases[i]))
            else:
                output.append(mnn_func2(x,self.weights[i],self.biases[i]))
            # print("ws",self.weights[i])
        # print("o",output[0].shape)
        # output = [mnn_func(x,ws,bs) for ws,bs in zip(self.weights,self.biases)]
        temp=np.hstack(output)
        final_output=np.expand_dims(temp,1)

        # self.iter_count=self.iter_count+1
        # print(self.iter_count)

        return final_output




    def _invert(self, data, input, mask, tag):
        """
        Inverse is... who knows!
        """
        # return npr.randn(data.shape[0], self.D)

        # temp=1.6*npr.rand(data.shape[0], self.D)-0.8
        # m=np.mean(data,axis=1).reshape(-1,1)
        # temp[:,0:1]=1.6*m/np.max(m)-0.8

        # temp=1*npr.rand(data.shape[0], self.D)+1
        # m=np.mean(data,axis=1).reshape(-1,1)
        # temp[:,0:1]=1*m/np.max(m)+1

        if tag is None:

            temp=1*npr.rand(data.shape[0], self.D)
            m=np.mean(data,axis=1).reshape(-1,1)
            temp[:,0:1]=1*m/np.max(m)

        else:
            temp=tag

        return temp


    def log_prior(self):
        self.iter_count=self.iter_count+1

        return 0

        # beta=100
        # return beta*np.sum(self.weights**2)+beta*np.sum(self.biases**2)
        # bs=[]
        # for i in range(len(self.biases)):
        #     for j in self.biases[i]:
        #         bs.append(np.sum(softplus(j)**2))
        # ws=[]
        # for i in range(len(self.weights)):
        #     for j in self.weights[i]:
        #         ws.append(np.sum(softplus(j)**2))
        # # print(bs)
        # return -beta*np.sum(ws)



            # smooth_spline_pens=[np.sum(np.abs(np.diff(np.diff(delta_ys[i],axis=0),axis=0))) for i in range(len(self.ps))]


# Allow general nonlinear emission models with neural networks
# class _MonotonicNeuralNetworkEmissions(_Emissions):
#     def __init__(self, N, K, D, M=0, hidden_layer_sizes=(50,), single_subspace=True):
#         assert single_subspace, "_NeuralNetworkEmissions only supports `single_subspace=True`"
#         super(_MonotonicNeuralNetworkEmissions, self).__init__(N, K, D, M=M, single_subspace=True)
#
#         # Initialize the neural network weights
#         assert N > D
#         layer_sizes = (D + M,) + hidden_layer_sizes + (N,)
#         self.weights = [npr.randn(m, n) * np.sqrt(1. / m) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
#         self.biases = [npr.randn(n) for n in layer_sizes[1:]]
#
#     @property
#     def params(self):
#         return self.weights, self.biases
#
#     @params.setter
#     def params(self, value):
#         self.weights, self.biases = value
#
#     def permute(self, perm):
#         pass
#
#     def forward(self, x, input, tag):
#         inputs = np.column_stack((x, input))
#         for W0, b0 in zip(self.weights, self.biases):
#             # print(W.shape)
#             W=softplus(W0)
#             b=softplus(b0)
#             outputs = np.dot(inputs, W) + b
#             inputs = np.tanh(outputs)
#         return outputs[:, None, :]
#
#     def _invert(self, data, input, mask, tag):
#         """
#         Inverse is... who knows!
#         """
#         # return npr.randn(data.shape[0], self.D)
#         temp=1.6*npr.rand(data.shape[0], self.D)-0.8
#         m=np.mean(data,axis=1).reshape(-1,1)
#         temp[:,0:1]=1.6*m/np.max(m)-0.8
#
#         return temp


# Allow general nonlinear emission models with neural networks
class _NeuralNetworkEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, hidden_layer_sizes=(50,), single_subspace=True):
        assert single_subspace, "_NeuralNetworkEmissions only supports `single_subspace=True`"
        super(_NeuralNetworkEmissions, self).__init__(N, K, D, M=M, single_subspace=True)

        # Initialize the neural network weights
        assert N > D
        layer_sizes = (D + M,) + hidden_layer_sizes + (N,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

    @property
    def params(self):
        return self.weights, self.biases

    @params.setter
    def params(self, value):
        self.weights, self.biases = value

    def permute(self, perm):
        pass

    def forward(self, x, input, tag):
        inputs = np.column_stack((x, input))
        for W, b in zip(self.weights, self.biases):
            # print(W.shape)
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs[:, None, :]

    def _invert(self, data, input, mask, tag):
        """
        Inverse is... who knows!
        """
        return npr.randn(data.shape[0], self.D)

# Allow general nonlinear emission models with neural networks
class _CompoundNeuralNetworkEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, hidden_layer_sizes=(50,), single_subspace=True, N_vec=None, D_vec=None, **kwargs):
        assert single_subspace, "_NeuralNetworkEmissions only supports `single_subspace=True`"
        super(_CompoundNeuralNetworkEmissions, self).__init__(N, K, D, M=M, single_subspace=True)


        # print(hidden_layer_sizes)
        #Make sure N_vec and D_vec are in correct form
        assert isinstance(N_vec, (np.ndarray, list, tuple))
        N_vec = np.array(N_vec, dtype=int)
        assert np.sum(N_vec) == N

        assert isinstance(D_vec, (np.ndarray, list, tuple)) and len(D_vec) == len(N_vec)
        D_vec = np.array(D_vec, dtype=int)
        assert np.sum(D_vec) == D

        self.N_vec, self.D_vec = N_vec, D_vec

        # Save the number of subpopulations
        self.P = len(self.N_vec)

        # The main purpose of this class is to wrap a bunch of emissions instances
        self.emissions_models = [_NeuralNetworkEmissions(n, K, d, hidden_layer_sizes=hidden_layer_sizes) for n, d in zip(N_vec, D_vec)]


    @property
    def params(self):
        return [em.params for em in self.emissions_models]

    @params.setter
    def params(self, value):
        assert len(value) == self.P
        for em, v in zip(self.emissions_models, value):
            em.params = v

    def permute(self, perm):
        pass

    def forward(self, x, input, tag):
        assert x.shape[1] == self.D
        D_offsets = np.cumsum(self.D_vec)[:-1]
        datas = []
        for em, xp in zip(self.emissions_models, np.split(x, D_offsets, axis=1)):
            datas.append(em.forward(xp, input, tag))
        return np.concatenate(datas, axis=2)


    def _invert(self, data, input, mask, tag):
        """
        Inverse is... who knows!
        """
        return npr.randn(data.shape[0], self.D)


# Observation models for SLDS
class _GaussianEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, ll_scale=0, **kwargs):
        super(_GaussianEmissionsMixin, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.ll_scale=ll_scale

    @property
    def params(self):
        return tuple(super(_GaussianEmissionsMixin, self).params) + (self.inv_etas,)

    @params.setter
    def params(self, value):
        self.inv_etas = value[-1]
        super(_GaussianEmissionsMixin, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(_GaussianEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        # print(data.shape)
        # print(np.sum(lls * mask[:, None, :], axis=2)/data.shape[0])
        # return np.sum(lls * mask[:, None, :], axis=2)
        if self.ll_scale==0:
            return np.sum(lls * mask[:, None, :], axis=2)
        #Scale log likelihood by number of data points (so trials w/ fewer data points are weighted equally per data point)
        else:
            # print(lls.shape)
            return np.sum(lls * mask[:, None, :], axis=2)*self.ll_scale/data.shape[0]

    @ensure_args_not_none
    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        return mus[np.arange(T), z, :] + np.sqrt(etas[z]) * npr.randn(T, self.N)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        return mus[:, 0, :] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)


# Observation models for SLDS
class _IsoGaussianEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, ll_scale=0, **kwargs):
        super(_IsoGaussianEmissionsMixin, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)
        self.inv_etas = -4 + npr.randn(1)
        self.ll_scale=ll_scale

    @property
    def params(self):
        return tuple(super(_IsoGaussianEmissionsMixin, self).params) + (self.inv_etas,)

    @params.setter
    def params(self, value):
        self.inv_etas = value[-1]
        super(_IsoGaussianEmissionsMixin, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(_IsoGaussianEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        # print(data.shape)
        # print(np.sum(lls * mask[:, None, :], axis=2)/data.shape[0])
        # return np.sum(lls * mask[:, None, :], axis=2)
        if self.ll_scale==0:
            return np.sum(lls * mask[:, None, :], axis=2)
        #Scale log likelihood by number of data points (so trials w/ fewer data points are weighted equally per data point)
        else:
            # print(lls.shape)
            # return np.sum(lls * mask[:, None, :], axis=2)*self.ll_scale/data.shape[0]/input[0,0]
            return np.sum(lls * mask[:, None, :], axis=2)*self.ll_scale/data.shape[0]

    @ensure_args_not_none
    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        return mus[np.arange(T), z, :] + np.sqrt(etas[z]) * npr.randn(T, self.N)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        return mus[:, 0, :] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)



class GaussianEmissions(_GaussianEmissionsMixin, _LinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

class GaussianCompoundEmissions(_GaussianEmissionsMixin, _CompoundLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class GaussianNonOrthCompoundEmissions(_GaussianEmissionsMixin, _CompoundLinearNonOrthEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class GaussianIdentityEmissions(_GaussianEmissionsMixin, _IdentityEmissions):
    pass


class GaussianNeuralNetworkEmissions(_GaussianEmissionsMixin, _NeuralNetworkEmissions):
    pass

class GaussianSplineEmissions(_GaussianEmissionsMixin, _SplineEmissions):
    pass

class GaussianLogisticEmissions(_GaussianEmissionsMixin, _LogisticEmissions):
    pass

class GaussianMonotonicNeuralNetworkEmissions(_GaussianEmissionsMixin, _MonotonicNeuralNetworkEmissions):
    pass

class IsotropicGaussianSplineEmissions(_IsoGaussianEmissionsMixin, _SplineEmissions):
    pass

class IsotropicGaussianLogisticEmissions(_IsoGaussianEmissionsMixin, _LogisticEmissions):
    pass

class IsotropicGaussianMonotonicNeuralNetworkEmissions(_IsoGaussianEmissionsMixin, _MonotonicNeuralNetworkEmissions):
    pass




class GaussianCompoundNeuralNetworkEmissions(_GaussianEmissionsMixin, _CompoundNeuralNetworkEmissions):
    pass

class _StudentsTEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_StudentsTEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_nus = np.log(4) * np.ones(1, N) if single_subspace else np.log(4) * np.ones(K, N)

    @property
    def params(self):
        return super(_StudentsTEmissionsMixin, self).params + (self.inv_etas, self.inv_nus)

    @params.setter
    def params(self, value):
        super(_StudentsTEmissionsMixin, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(_StudentsTEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]
            self.inv_nus = self.inv_nus[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        N, etas, nus = self.N, np.exp(self.inv_etas), np.exp(self.inv_nus)
        mus = self.forward(x, input, tag)

        resid = data[:, None, :] - mus
        z = resid / etas
        return -0.5 * (nus + N) * np.log(1.0 + (resid * z).sum(axis=2) / nus) + \
            gammaln((nus + N) / 2.0) - gammaln(nus / 2.0) - N / 2.0 * np.log(nus) \
            -N / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(etas), axis=1)

    @ensure_args_not_none
    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        taus = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        return mus[np.arange(T), z, :] + np.sqrt(etas[z] / taus) * npr.randn(T, self.N)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)


class StudentsTEmissions(_StudentsTEmissionsMixin, _LinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class StudentsTIdentityEmissions(_StudentsTEmissionsMixin, _IdentityEmissions):
    pass


class StudentsTNeuralNetworkEmissions(_StudentsTEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _BernoulliEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="logit", **kwargs):
        super(BernoulliEmissions, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        mean_functions = dict(
            logit=logistic
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            logit=logit
            )
        self.link = link_functions[link]

    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == int and data.min() >= 0 and data.max() <= 1
        ps = self.mean(self.forward(x, input, tag))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = data[:, None, :] * np.log(ps) + (1 - data[:, None, :]) * np.log(1 - ps)
        return np.sum(lls * mask[:, None, :], axis=2)

    @ensure_args_not_none
    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, .1, .9))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        ps = self.mean(self.forward(x, input, tag))
        return npr.rand(T, self.N) < ps[np.arange(T), z,:]

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        ps = self.mean(self.forward(variational_mean, input, tag))
        return ps[:,0,:] if self.single_subspace else np.sum(ps * expected_states[:,:,None], axis=1)


class BernoulliEmissions(_BernoulliEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(logits, inputs=inputs, masks=masks, tags=tags)


class BernoulliIdentityEmissions(_BernoulliEmissionsMixin, _IdentityEmissions):
    pass


class BernoulliNeuralNetworkEmissions(_BernoulliEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _PoissonEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="log", **kwargs):
        super(_PoissonEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        mean_functions = dict(
            log=lambda x: np.exp(x),
            softplus=softplus
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            log=lambda rate: np.log(rate),
            softplus=inv_softplus
            )
        self.link = link_functions[link]

    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == int
        lambdas = self.mean(self.forward(x, input, tag))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = -gammaln(data[:,None,:] + 1) -lambdas + data[:,None,:] * np.log(lambdas)
        return np.sum(lls * mask[:, None, :], axis=2)

    @ensure_args_not_none
    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, .1, np.inf))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        lambdas = self.mean(self.forward(x, input, tag))
        y = npr.poisson(lambdas[np.arange(T), z, :])
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        lambdas = self.mean(self.forward(variational_mean, input, tag))
        return lambdas[:,0,:] if self.single_subspace else np.sum(lambdas * expected_states[:,:,None], axis=1)

class PoissonEmissions(_PoissonEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

class PoissonCompoundEmissions(_PoissonEmissionsMixin, _CompoundLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

class PoissonIdentityEmissions(_PoissonEmissionsMixin, _IdentityEmissions):
    pass


class PoissonNeuralNetworkEmissions(_PoissonEmissionsMixin, _NeuralNetworkEmissions):
    pass

class PoissonCompoundNeuralNetworkEmissions(_PoissonEmissionsMixin, _CompoundNeuralNetworkEmissions):
    pass

class _AutoRegressiveEmissionsMixin(object):
    """
    Include past observations as a covariate in the SLDS emissions.
    The AR model is restricted to be diagonal.
    """
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_AutoRegressiveEmissionsMixin, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)

        # Initialize AR component of the model
        self.As = npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def params(self):
        return super(_AutoRegressiveEmissionsMixin, self).params + (self.As, self.inv_etas)

    @params.setter
    def params(self, value):
        self.As, self.inv_etas = value[-2:]
        super(_AutoRegressiveEmissionsMixin, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(_AutoRegressiveEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.As = self.inv_nus[perm]
            self.inv_etas = self.inv_etas[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.forward(x, input, tag)
        pad = np.zeros((1, 1, self.N)) if self.single_subspace else np.zeros((1, self.K, self.N))
        mus = mus + np.concatenate((pad, self.As[None, :, :] * data[:-1, None, :]))

        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        return np.sum(lls * mask[:, None, :], axis=2)

    @ensure_args_not_none
    def invert(self, data, input=None, mask=None, tag=None):
        pad = np.zeros((1, 1, self.N)) if self.single_subspace else np.zeros((1, self.K, self.N))
        resid = data - np.concatenate((pad, self.As[None, :, :] * data[:-1, None, :]))
        return self._invert(resid, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T, N = z.shape[0], self.N
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)

        y = np.zeros((T, N))
        y[0] = mus[0, z[0], :] + np.sqrt(etas[z[0]]) * npr.randn(N)
        for t in range(1, T):
            y[t] = mus[t, z[t], :] + self.As[z[t]] * y[t-1] + np.sqrt(etas[z[0]]) * npr.randn(N)
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        mus[1:] += self.As[None, :, :] * data[:-1, None, :]
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states, axis=1)


class AutoRegressiveEmissions(_AutoRegressiveEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        # Initialize the subspace with PCA
        from sklearn.decomposition import PCA
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]

        # Solve a linear regression for the AR coefficients.
        from sklearn.linear_model import LinearRegression
        for n in range(self.N):
            lr = LinearRegression()
            lr.fit(np.concatenate([d[:-1, n] for d in datas])[:,None],
                   np.concatenate([d[1:, n] for d in datas]))
            self.As[:,n] = lr.coef_[0]

        # Compute the residuals of the AR model
        pad = np.zeros((1,self.N))
        mus = [np.concatenate((pad, self.As[0] * d[:-1])) for d in datas]
        residuals = [data - mu for data, mu in zip(datas, mus)]

        # Run PCA on the residuals to initialize C and d
        pca = self._initialize_with_pca(residuals, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class AutoRegressiveIdentityEmissions(_AutoRegressiveEmissionsMixin, _IdentityEmissions):
    pass


class AutoRegressiveNeuralNetworkEmissions(_AutoRegressiveEmissionsMixin, _NeuralNetworkEmissions):
    pass
