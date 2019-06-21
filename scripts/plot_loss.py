import os
import sys
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

fig, ax1 = plt.subplots()

loss = None
try:
    loss = np.loadtxt(sys.argv[1],skiprows=1, dtype=float, delimiter=',')
except Exception as e:
    print('failed. trying with ;')
    loss = np.loadtxt(sys.argv[1],skiprows=1, dtype=float, delimiter=';')

color = 'tab:red'
l1 = ax1.plot(loss[:,1], label='training loss', color=color)
l2 = ax1.plot(loss[:,-1], label='val. loss', color='purple')
ax1.set_xlabel('epoch nr.')
ax1.set_ylabel('loss')
ax1.set_yscale('log')

color = 'tab:blue'
ax2 = ax1.twinx()
l3 = ax2.plot(loss[:,2], label='learn. rate', color=color)
ax2.set_ylabel('l.r.', color=color)

lns = l1+l2+l3
labs = [l.get_label() for l in lns]
plt.legend(lns, labs)


plt.title('Loss and l.r. for model')
fig.tight_layout() 

name = sys.argv[1].split('.csv')[:-1][0]
name = name + ".png"
print(name)


plt.gcf().savefig(name, bb_inches='tight')
