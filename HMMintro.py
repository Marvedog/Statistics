# HMMs 
from pomegranate import *
import matplotlib.pyplot as plt
import timeit
import numpy as np
import pygraphviz as pgv


seq = list('CGACTACTGACTACTCGCCGACGCGACTGCCGTCTATACTGCGCATACGGC')

d1 = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})
d2 = DiscreteDistribution({'A': 0.10, 'C': 0.40, 'G': 0.40, 'T': 0.10})

# General Mixture Model Setup
gmm = GeneralMixtureModel( [d1, d2] )

# Hidden Markov Model Setup
s1 = State( d1, name='background' )
s2 = State( d2, name='CG island' )

hmm = HiddenMarkovModel('Island Finder')
hmm.add_states(s1, s2)
hmm.add_transition( hmm.start, s1, 0.5 )
hmm.add_transition( hmm.start, s2, 0.5 )
hmm.add_transition( s1, s1, 0.5 )
hmm.add_transition( s1, s2, 0.5 )
hmm.add_transition( s2, s1, 0.5 )
hmm.add_transition( s2, s2, 0.5 )
hmm.bake()

plt.figure( figsize=(10,6) )
hmm.plot()
plt.show()


gmm_predictions = gmm.predict( np.array(seq) )
hmm_predictions = hmm.predict( seq )

print "sequence: {}".format( ''.join( seq ) )
print "gmm pred: {}".format( ''.join( map( str, gmm_predictions ) ) )
print "hmm pred: {}".format( ''.join( map( str, hmm_predictions ) ) )

hmm = HiddenMarkovModel()
hmm.add_states(s1, s2)
hmm.add_transition( hmm.start, s1, 0.5 )
hmm.add_transition( hmm.start, s2, 0.5 )
hmm.add_transition( s1, s1, 0.9 )
hmm.add_transition( s1, s2, 0.1 )
hmm.add_transition( s2, s1, 0.1 )
hmm.add_transition( s2, s2, 0.9 )
hmm.bake()

plt.figure( figsize=(10,6) )
hmm.plot()
plt.show()
hmm_predictions = hmm.predict( seq )

print "sequence: {}".format( ''.join( seq ) )
print "gmm pred: {}".format( ''.join( map( str, gmm_predictions ) ) )
print "hmm pred: {}".format( ''.join( map( str, hmm_predictions ) ) )
print
print "hmm state 0: {}".format( hmm.states[0].name )
print "hmm state 1: {}".format( hmm.states[1].name )

print hmm.predict_proba( seq )
