import numpy as np

alpha = 737
s0 = 0.188
s1 = 0.064
LM = np.array([[0,alpha,2*alpha],[s0,0,0],[0,s1,0]]) # leslie matrix
eigenvalues, eigenvectors = np.linalg.eig(LM)
