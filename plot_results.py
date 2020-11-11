import numpy as np
import matplotlib.pyplot as plt

labels = ['svm', 'knn', 'log_reg', 'gau_nb', 'rf', 'lda']
# dtw=[0.644,0.573,0.480,0.578,0.658,0.613]
# euc_dist=[0.627,0.547,0.564,0.569,0.622,0.547]
# dtw=[0.875,0.762,0.850,0.863,0.906,0.875]
# euc_dist=[0.831,0.706,0.787,0.787,0.831,0.758]

dtw=[0.917,0.779,0.729,0.879,0.878,0.805]
euc_dist=[0.978,0.919,0.943,0.963,0.943,0.920]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, dtw, width, label='dtw')
rects2 = ax.bar(x + width / 2, euc_dist, width, label='euc')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0,1)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()