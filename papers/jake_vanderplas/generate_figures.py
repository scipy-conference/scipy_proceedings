import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import erfinv


############################################################
# Photon Counts
print("Example #1")

# Generate the data
np.random.seed(2)  # for repeatability
e = np.random.normal(30, 3, 50)
F = np.random.normal(1000, e)

# Frequentist Result
w = 1. / e ** 2
print("""
      F_true = {0}
      F_est  = {1:.0f} +/- {2:.0f} (based on {3} measurements)
      """.format(1000, (w * F).sum() / w.sum(), w.sum() ** -0.5, len(F)))


######################################################################
# Linear Model Example

#----------------------------------------------------------------------
# Define the data
np.random.seed(42)
theta_true = (25, 0.5)
xdata = 100 * np.random.random(20)
ydata = theta_true[0] + theta_true[1] * xdata
ydata = np.random.normal(ydata, 10)

#----------------------------------------------------------------------
# Compute the frequentist version
X = np.vstack([np.ones_like(xdata), xdata]).T
theta_freq = np.linalg.solve(np.dot(X.T, X),
                             np.dot(X.T, ydata))
y_model = np.dot(X, theta_freq)
sigma_y = np.std(ydata - y_model)
Sigma_freq = sigma_y ** 2 * np.linalg.inv(np.dot(X.T, X))

print("Frequentist Version")
print("  theta = {0}".format(theta_freq))
print("  sigma_y = {0}".format(sigma_y))
print("  Sigma = {0}".format(Sigma_freq))

#----------------------------------------------------------------------
# Fit with emcee
import emcee
print("Emcee version {0}".format(emcee.__version__))

# Define our posterior using Python functions
# for clarity, I've separated-out the prior and likelihood
# but this is not necessary. Note that emcee requires log-posterior

def log_prior(theta):
    alpha, beta, sigma = theta
    if sigma < 0:
        return -np.inf  # log(0)
    else:
        return -1.5 * np.log(1 + beta ** 2) - np.log(sigma)

def log_likelihood(theta, x, y):
    alpha, beta, sigma = theta
    y_model = alpha + beta * x
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_posterior(theta, x, y):
    return log_prior(theta) + log_likelihood(theta, x, y)

# Here we'll set up the computation. emcee combines multiple "walkers",
# each of which is its own MCMC chain. The number of trace results will
# be nwalkers * nsteps

ndim = 3  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nburn = 1000  # "burn-in" period to let chains stabilize
nsteps = 2000  # number of MCMC steps to take

# set theta near the maximum likelihood, with 
np.random.seed(0)
starting_guesses = np.random.random((nwalkers, ndim))

# Here's the function call where all the work happens:
# we'll time it using IPython's %time magic

print("  running emcee sampler...")
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                args=[xdata, ydata])
sampler.run_mcmc(starting_guesses, nsteps)
emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
print("  done")


#----------------------------------------------------------------------
# Fit with PyMC

import pymc
print("PyMC version: {0}".format(pymc.__version__))

# Define the variables needed for the routine, with their prior distributions
alpha = pymc.Uniform('alpha', -100, 100)

@pymc.stochastic(observed=False)
def beta(value=0):
    return -1.5 * np.log(1 + value ** 2)

@pymc.stochastic(observed=False)
def sigma(value=1):
    return -np.log(abs(value))

# Define the form of the model and likelihood
@pymc.deterministic
def y_model(x=xdata, alpha=alpha, beta=beta):
    return alpha + beta * x

y = pymc.Normal('y', mu=y_model, tau=1. / sigma ** 2, observed=True, value=ydata)

# package the full model in a dictionary
model1 = dict(alpha=alpha, beta=beta, sigma=sigma,
              y_model=y_model, y=y)

# run the basic MCMC: we'll do 100000 iterations to match emcee above
S = pymc.MCMC(model1)
S.sample(iter=100000, burn=50000)

# extract the traces and plot the results
pymc_trace = [S.trace('alpha')[:],
              S.trace('beta')[:],
              S.trace('sigma')[:]]


#----------------------------------------------------------------------
# PyStan solution
import pystan
print("PyStan version: {0}".format(pystan.__version__))
# Create the Stan model
#  this is done by defining a string of Stan code.

fit_code = """
data {
    int<lower=0> N; // number of points
    real x[N]; // x values
    real y[N]; // y values
}

parameters {
    real alpha_perp;
    real<lower=-pi()/2, upper=pi()/2> theta;
    real log_sigma;
}

transformed parameters {
    real alpha;
    real beta;
    real sigma;
    real ymodel[N];
    
    alpha <- alpha_perp / cos(theta);
    beta <- sin(theta);
    sigma <- exp(log_sigma);
    for (j in 1:N)
    ymodel[j] <- alpha + beta * x[j];
}

model {
    y ~ normal(ymodel, sigma);
}
"""

# perform the fit
fit_data = {'N': len(xdata), 'x': xdata, 'y': ydata}
fit = pystan.stan(model_code=fit_code, data=fit_data, iter=25000, chains=4)

# extract the traces
traces = fit.extract()
pystan_trace = [traces['alpha'], traces['beta'], traces['sigma']]

#----------------------------------------------------------------------
# Visualize the results
# Create some convenience routines for plotting

def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    
    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
    """Plot traces and contours"""
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace[0], trace[1], ',k', alpha=0.1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')


# compute the ellipse pricipal axes and rotation from covariance
def get_principal(Sigma):
    # See Ivezic, Connolly, VanderPlas, and Gray, section 3.5.2
    sigma_x2 = Sigma[0, 0]
    sigma_y2 = Sigma[1, 1]
    sigma_xy = Sigma[0, 1]

    alpha = 0.5 * np.arctan2(2 * sigma_xy, sigma_x2 - sigma_y2)
    tmp1 = 0.5 * (sigma_x2 + sigma_y2)
    tmp2 = np.sqrt(0.25 * (sigma_x2 - sigma_y2) ** 2 + sigma_xy ** 2)

    return np.sqrt(tmp1 + tmp2), np.sqrt(tmp1 - tmp2), alpha

fig, ax = plt.subplots(figsize=(6, 6))

plot_MCMC_trace(ax, xdata, ydata, emcee_trace, True,
                colors='blue', linewidths=2)
plot_MCMC_trace(ax, xdata, ydata, pymc_trace,
                colors='red', linewidths=2)
plot_MCMC_trace(ax, xdata, ydata, pystan_trace,
                colors='green', linewidths=2)

# plot frequentist ellipses
from matplotlib.patches import Ellipse

sigma1, sigma2, alpha = get_principal(Sigma_freq)
for level in [0.683, 0.955]:
    # 2 dimensions: we need frac^2 = level
    # in order to compare directly with MCMC contours
    nsigma_eff = np.sqrt(2) * erfinv(np.sqrt(level))
    ax.add_patch(Ellipse(theta_freq,
                         2 * nsigma_eff * sigma1, 2 * nsigma_eff * sigma2,
                         angle=np.degrees(alpha),
                         lw=2, ec='k', fc='none'))

ax.plot([0, 0], [0, 0], '-k', lw=2)

ax.legend(ax.lines[-1:] + ax.collections[::2],
          ['frequentist', 'emcee', 'pymc', 'pystan'], fontsize=14)

ax.set_xlim(10, 40)
ax.set_ylim(0.2, 0.7)

print("saving figure1.png")
fig.savefig("figure1.png")
plt.show()
