import numpy as np
import matplotlib.pyplot as plt

s = 'sarsa_vtg'

data = np.load('./scores/'+s+'.npy')

plt.figure(figsize=(5,4))
plt.xlabel('epochs')
plt.ylabel('score')
plt.plot(np.arange(1,2000+1), data)
plt.savefig('./plots/'+s+'_plt.png')
plt.show()


plt.figure(figsize=(5,4))
plt.xlabel('scores')
plt.ylabel('count')
plt.hist(data, bins=20)
plt.savefig('./plots/'+s+'_hist.png')
plt.show()
