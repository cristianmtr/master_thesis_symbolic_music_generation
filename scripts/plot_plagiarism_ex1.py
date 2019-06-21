import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

sizes = [15, 30, 45, 60, 100]
scores = [0.683, 0.573, 0.278, 0.126, 0.131]

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.figure(figsize=(7,7))
plt.scatter(sizes,scores, c='magenta')
plt.ylabel('plagiarim scores', fontsize=18)
plt.xlabel('dataset size (k)', fontsize=18)
plt.title('Experiment 1: Dataset sizes and plagiarism scores', fontsize=18)
plt.gcf().savefig('ex1_scores.png',bbox_inches='tight')
