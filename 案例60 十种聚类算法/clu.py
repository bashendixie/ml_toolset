from numpy import where
from matplotlib import pyplot
from numpy import unique
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture

def make_data():
	# define dataset
	X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
	# create scatter plot for samples from each class
	for class_value in range(2):
		# get row indexes for samples with this class
		row_ix = where(y == class_value)
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
	# show the plot
	pyplot.show()

def use_AffinityPropagation():
	# define dataset
	X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
							   random_state=4)
	# define the model
	model = AffinityPropagation(damping=0.9)
	# fit the model
	model.fit(X)
	# assign a cluster to each example
	yhat = model.predict(X)
	# retrieve unique clusters
	clusters = unique(yhat)
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
	# show the plot
	pyplot.show()

def use_AgglomerativeClustering():
	# define dataset
	X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
							   random_state=4)
	# define the model
	model = AgglomerativeClustering(n_clusters=2)
	# fit model and predict clusters
	yhat = model.fit_predict(X)
	# retrieve unique clusters
	clusters = unique(yhat)
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
	# show the plot
	pyplot.show()

def use_Birch():
	# define dataset
	X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
							   random_state=4)
	# define the model
	model = Birch(threshold=0.01, n_clusters=2)
	# fit the model
	model.fit(X)
	# assign a cluster to each example
	yhat = model.predict(X)
	# retrieve unique clusters
	clusters = unique(yhat)
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
	# show the plot
	pyplot.show()

def use_DBSCAN():
	# define dataset
	X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
							   random_state=4)
	# define the model
	model = DBSCAN(eps=0.30, min_samples=9)
	# fit model and predict clusters
	yhat = model.fit_predict(X)
	# retrieve unique clusters
	clusters = unique(yhat)
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
	# show the plot
	pyplot.show()

def use_kmeans():
	# define dataset
	X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
							   random_state=4)
	# define the model
	model = KMeans(n_clusters=2)
	# fit the model
	model.fit(X)
	# assign a cluster to each example
	yhat = model.predict(X)
	# retrieve unique clusters
	clusters = unique(yhat)
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
	# show the plot
	pyplot.show()

def use_mini_kmeans():
	# define dataset
	X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
							   random_state=4)
	# define the model
	model = MiniBatchKMeans(n_clusters=2)
	# fit the model
	model.fit(X)
	# assign a cluster to each example
	yhat = model.predict(X)
	# retrieve unique clusters
	clusters = unique(yhat)
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
	# show the plot
	pyplot.show()

def use_MeanShift():
	# define dataset
	X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
							   random_state=4)
	# define the model
	model = MeanShift()
	# fit model and predict clusters
	yhat = model.fit_predict(X)
	# retrieve unique clusters
	clusters = unique(yhat)
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
	# show the plot
	pyplot.show()

def use_OPTICS():
	# define dataset
	X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
							   random_state=4)
	# define the model
	model = OPTICS(eps=0.8, min_samples=10)
	# fit model and predict clusters
	yhat = model.fit_predict(X)
	# retrieve unique clusters
	clusters = unique(yhat)
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
	# show the plot
	pyplot.show()

def use_SpectralClustering():
	# define dataset
	X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
							   random_state=4)
	# define the model
	model = SpectralClustering(n_clusters=2)
	# fit model and predict clusters
	yhat = model.fit_predict(X)
	# retrieve unique clusters
	clusters = unique(yhat)
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
	# show the plot
	pyplot.show()


def use_GaussianMixture():
	# define dataset
	X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
							   random_state=4)
	# define the model
	model = GaussianMixture(n_components=2)
	# fit the model
	model.fit(X)
	# assign a cluster to each example
	yhat = model.predict(X)
	# retrieve unique clusters
	clusters = unique(yhat)
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
	# show the plot
	pyplot.show()

use_GaussianMixture()