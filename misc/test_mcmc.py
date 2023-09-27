import numpy as np
import scipy.stats
import chaospy

f = lambda x: x**2
Q = lambda t1, t0: np.exp(-(t1 - t0)**2/2)/(np.sqrt(2 * np.pi)) # random walk

std = 0.1
P = lambda x: np.exp(-0.5*((0.5 - f(x))**2)/std) # likelihood
U = scipy.stats.uniform # prior

x0 = 0
xt = x0
samples = []
for i in range(10000):
    xt_candidate = np.random.normal(xt, 1)
    accept_prob = (P(xt_candidate) * U.pdf(xt_candidate) * Q(xt, xt_candidate))/(P(xt) * U.pdf(xt) * Q(xt_candidate, xt))
    if np.random.uniform(0, 1) < accept_prob:
        xt = xt_candidate
    samples.append(xt)
    
burn_in = 1000
samples = np.array(samples[burn_in:])

## PCE
U2 = chaospy.Uniform(0,1)
J = chaospy.J(U2)

gauss_quads = chaospy.generate_quadrature(7, J, rule="gaussian") #(order of quadrature, joint prob, rule)
nodes, weights = gauss_quads
expansion = chaospy.generate_expansion(5, J)

evals = f(nodes[0,:])
model = chaospy.fit_quadrature(expansion, nodes, weights, evals)

P2 = lambda x: np.exp(-0.5*((0.5 - model(x))**2)/std) # likelihood using PCE model

x0 = 0
xt = x0
samples2 = []
for i in range(10000):
    xt_candidate = np.random.normal(xt, 1)
    accept_prob = (P2(xt_candidate) * U2.pdf(xt_candidate) * Q(xt, xt_candidate))/(P2(xt) * U2.pdf(xt) * Q(xt_candidate, xt))
    if np.random.uniform(0, 1) < accept_prob:
        xt = xt_candidate
    samples2.append(xt)
    
burn_in = 1000
samples2 = np.array(samples2[burn_in:])


import matplotlib.pyplot as plt
xs = np.arange(0, 1, 0.01)
ps = P(xs) * U.pdf(xs) # unnormalized posterior
pcs = P2(xs) * U.pdf(xs) # unnormalized posterior, using PCE model

plt.figure()
plt.hist(samples, density=True, alpha=0.5, bins=100)
plt.hist(samples2, density=True, alpha=0.5, bins=100)
plt.plot(xs, ps, linestyle="--", label="MC")
plt.plot(xs, pcs, linestyle="-.", label="PC")
# plt.hist(pcs, density=True, bins=100)
plt.legend()
plt.savefig("test_mcmc.png")


samples_prio = U2.sample(9000)

fig, ax = plt.subplots(2, 1)
ax[0].set_title("Prior")
ax[0].hist([f(samples_prio), model(samples_prio)], density=True, alpha=0.5, bins=100, label="MC prior")
# ax[0].hist(, density=True, alpha=0.5, bins=100, label="PC prior")

ax[1].set_title("Posterior")
ax[1].hist([f(samples), model(samples2)], density=True, alpha=0.5, bins=100, label="MC post")
# ax[1].hist(, density=True, alpha=0.5, bins=100, label="PC post")

plt.savefig("test_mcmc_output.png")