import numpy as np
import matplotlib.pyplot as plt

s = np.loadtxt('sims.tsv')
s = np.loadtxt('sims.tsv',delimiter='\t', dtype=object)
sims = s[:,-1]
sims = sims.astype('float32')
plt.hist(sims)
plt.show()
np.mean(sims)
np.std(sims)
