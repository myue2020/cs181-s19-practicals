import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

s = 'q_vtg'

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

# etas = [.01,.01,.01,.05,.05,.05,.1,.1,.1]
# gammas = [.6,.7,.8,.6,.7,.8,.6,.7,.8]
# maxes = [12,14,21,34,117,80,36,74,97]
# avgs = [1.28,0.413,0.52,2.95,14.853,7.88,4.64,12.88,12.39]
#
# fig = plt.figure(figsize=(5,4))
# bar = fig.add_subplot(111, projection='3d')
# bar.bar3d(etas,gammas,0,.01,.05,maxes)
# plt.xlabel('eta')
# plt.ylabel('gamma')
# bar.set_title('max score')
# plt.savefig('./plots/tuning_max.png')
# # plt.show()
#
#
#
#
# etas = [.01,.01,.01,.05,.05,.05,.1,.1,.1]
# gammas = [.6,.7,.8,.6,.7,.8,.6,.7,.8]
# maxes = [12,14,21,34,117,80,36,74,97]
# avgs = [1.28,0.413,0.52,2.95,14.853,7.88,4.64,12.88,12.39]
#
# fig = plt.figure(figsize=(5,4))
# bar2 = fig.add_subplot(111, projection='3d')
# bar2.bar3d(etas,gammas,0,.01,.05,avgs)
# bar2.set_title('avg score')
# plt.xlabel('eta')
# plt.ylabel('gamma')
# plt.savefig('./plots/tuning_avg.png')
# # plt.show()
#
#
#
#
# dng = np.load('./scores/cubic.npy')
# print(max(dng))
# print(np.mean(dng[3*1000//4:1000]))
