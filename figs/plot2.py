import matplotlib.pyplot as plt
# x = [2,4,6,8,10,12]

x = ['SVM', 'KNN', 'Log_reg', 'Gau_NB', 'RF', 'LDA']

# k1 = [0.538,0.613,0.613,0.676,0.644,0.636] #mds1
# k2 = [0.391,0.591,0.609,0.613,0.596,0.573]

# k1 = [0.848,0.868,0.904,0.897,0.886,0.890] #mds2
# k2 = [0.886,0.902,0.911,0.912,0.912,0.911]

k1=[0.926,0.779,0.788,0.879,0.878,0.805] #dtw-half
k2=[0.926,0.760,0.774,0.895,0.890,0.776]

# k1=[0.978,0.919,0.943,0.963,0.943,0.920] #euc-half
# k2=[0.964,0.888,0.927,0.951,0.924,0.906]

plt.plot(x,k1,'s-',label="Full")
plt.plot(x,k2,'o-',label="Half")
plt.xlabel("Components")
plt.ylabel("Accuracy")
plt.ylim(0.5,1)
plt.legend(loc = "best")
plt.savefig('/media/volume/sh/DTW_MDS/figs/fig_half1.png')
plt.show()
a=0