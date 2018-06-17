# Discrete densities 
from pomegranate import *
import matplotlib.pyplot as plt
import timeit
import numpy as np

d = DiscreteDistribution({'A': 0.1, 'C': 0.25, 'G': 0.50, 'T': 0.15})


print "P({}|M) = {:.3}".format( 'A', np.e ** d.log_probability( 'A' ) )
print "P({}|M) = {:.3}".format( 'G', np.e ** d.log_probability( 'G' ) )
print "P({}|M) = {:.3}".format( '?', np.e ** d.log_probability( '?' ) )

d.fit( list('CAGCATCATCATCATAGCACCATAGAAAGATAAAAT') )
print d.parameters