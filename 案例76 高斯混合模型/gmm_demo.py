import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture as GMM

plt.style.use('seaborn')

# 特定分布的一维数据
np.random.seed(2)
x = np.concatenate([np.random.normal(0, 2, 2000), np.random.normal(5, 5, 2000), np.random.normal(3, 0.5, 600)])
plt.hist(x, 80)
plt.xlim(-10, 20);
plt.show()

# 高斯混合模型将允许我们近似这个密度：
X = x[:, np.newaxis]
clf = GMM(4, max_iter=500, random_state=3).fit(X)
xpdf = np.linspace(-10, 20, 1000)
density = np.array([np.exp(clf.score([[xp]])) for xp in xpdf])

plt.hist(x, 80, density=True, alpha=0.5)
plt.plot(xpdf, density, '-r')
plt.xlim(-10, 20)
plt.show()

plt.hist(x, 80, alpha=0.3)
plt.plot(xpdf, density, '-r')
for i in range(clf.n_components):
    pdf = clf.weights_[i] * stats.norm(clf.means_[i, 0], np.sqrt(clf.covariances_[i, 0])).pdf(xpdf)
    plt.fill(xpdf, pdf, facecolor='gray', edgecolor='none', alpha=0.3)
plt.xlim(-10, 20)
plt.show()


np.random.seed(0)

# Add 20 outliers
true_outliers = np.sort(np.random.randint(0, len(x), 20))
y = x.copy()
y[true_outliers] += 50 * np.random.randn(20)

clf = GMM(4, max_iter=500, random_state=0).fit(y[:, np.newaxis])
xpdf = np.linspace(-10, 20, 1000)
density_noise = np.array([np.exp(clf.score([[xp]])) for xp in xpdf])

plt.hist(y, 80, density=True, alpha=0.5)
plt.plot(xpdf, density_noise, '-r')
plt.xlim(-15, 30);
plt.show()

log_likelihood = np.array([clf.score_samples([[yy]]) for yy in y])
# log_likelihood = clf.score_samples(y[:, np.newaxis])[0]
plt.plot(y, log_likelihood, '.k');
plt.show()

detected_outliers = np.where(log_likelihood < -9)[0]

print("true outliers:")
print(true_outliers)
print("\ndetected outliers:")
print(detected_outliers)

