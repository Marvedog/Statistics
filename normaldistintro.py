# Basic of the normal distribution

from pomegranate import *
import matplotlib.pyplot as plt
import timeit
import numpy as np

# A signle univariate normal
d = NormalDistribution( 0, 1 ) # The normal distribution

# Fit model to normally distirbuted data
data = np.random.randn(100) * 0.9 + 0.4
d.fit( data )

print data.mean() 
print data.std()

plt.figure( figsize=(10,6) )
d.plot( n=10000, edgecolor='c', facecolor='c', alpha=1, bins=25 )
plt.show()

# MUltivariate gaussian

d1 = MultivariateGaussianDistribution( np.arange(5) * 5, np.eye(5) )
print d1.sample()

data = np.random.randn(1000, 5) + np.arange(5) * 8

d1.fit(data)
print "mu: [{:.5}, {:.5}, {:.5}, {:.5}, {:.5}]".format( *d1.parameters[0] )
print "cov: \n {}".format( np.array(d1.parameters[1]))

# Gaussian mixture models
data2 = np.array([np.concatenate( (np.random.randn(1000) * 2.75 + 1.25, np.random.randn(2000) * 1.2 + 7.85) )]).T

weights = np.array([0.33, 0.67])
d2 = GeneralMixtureModel( [NormalDistribution(2, 1), NormalDistribution(8, 1)], weights=weights )

labels = d2.predict( data2 )
print "{} 1 labels, {} 0 labels".format( labels.sum(), labels.shape[0] - labels.sum())

labels = d2.predict_proba( data2 )
print labels[:5]
print "{} 0 labels, {} 1 labels".format( *labels.sum(axis=0) )

d2.fit( data, verbose=True )

labels = d2.predict( data2 )

print "Hard Classification"
print "{} 0 labels, {} 1 labels".format( labels.shape[0] - labels.sum(), labels.sum() )

print
print "Soft Classification"
labels = d2.predict_proba( data2 )
print "{} 0 labels, {} 1 labels".format( *labels.sum(axis=0) )