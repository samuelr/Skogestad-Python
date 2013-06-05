"""
@author: Samuel
"""
import numpy as np
import matplotlib.pyplot as plt

I = np.identity(2)
def G(s):
        """process transfer matrix"""
        return 1 / (75*s +1) * np.matrix([[87.8, -86.4],
                                          [108.2,-109.6]])
def K(s):
    """controller"""
    return (0.7/s)*np.linalg.inv((G(s)))
                                 
def l(s): return (0.7/s)
def e(s): return 1/(1+l(s))
def t(s):return 1-e(s)
def L(s):return l(s)*K(s)*G(s)
def S(s):return e(s)*I
def T(s):return t(s)*I
def w_I(s):
    """uncertainty"""
    return (s + 0.2) / (0.5 * s + 1)
def W_I(s):
    """uncertainty weights"""
    return w_I(s)*np.matrix([[1, 0], 
                             [0, 1]])
def w_p(s):return (s/2 + 0.05) /s
def W_p(s):return w_p(s)*np.matrix([[1, 0], 
                             [0, 1]])

def sigmas(A):
    """Return the singular values of A"""
    return np.linalg.svd(A, compute_uv=False)

def specrad_w_I_T_I(s):
    '''check for RS with w_I_T'''
    return np.abs(w_I(s)*t(s))

frequency = np.logspace(-3, 2, 1000)
s = 1j * frequency

maxsigmaN22s= [np.max(sigmas(w_p(si)*S(si))) for si in s]

specrad_w_I_T_Is=[specrad_w_I_T_I(si) for si in s]  
   
specrad_N=[(np.abs(np.max(sigmas(w_I(si)*T(si))) + np.max(sigmas(w_p(si)*S(si)))))*
           (1+np.sqrt(((np.max(sigmas(K(si)))))/(np.min(sigmas(K(si)))))) for si in s]

specrad_N=[np.sqrt( np.power(np.abs(w_p(si)*e(si)),2) + np.power(np.abs(w_I(si)*t(si)),2) +
                    np.abs(w_p(si)*e(si)) * np.abs(w_I(si)*t(si)) *
                    (   (((np.max(sigmas(K(si)))))/(np.min(sigmas(K(si))))) +
                        1/(((np.max(sigmas(K(si)))))/(np.min(sigmas(K(si))))) ) )  for si in s ]                   

plt.semilogx(frequency, maxsigmaN22s, 'b--',
             frequency, specrad_w_I_T_Is, 'r--',
             frequency, specrad_N, 'k')

plt.legend(('NP','RS','RP'), 'best', shadow=False)
plt.xlabel('frequency')
plt.ylabel('structured singular value (mu)')
plt.show()
