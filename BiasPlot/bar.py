import sys, os
import numpy as np
import matplotlib.pyplot as plt

n_groups = 6

sources = ['CNN', 'The New York Times', 'The Atlantic', 'Fox News', ' The Guardian', 'The Washington Post']
op = [47.894, 54.291, 44.106, 39.779, 40.567, 44.071]
non_op = [74.637, 79.284, 69.653, 67.630, 74.881, 75.626]

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, op, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='OpEds')

rects2 = ax.bar(index + bar_width, non_op, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='Other')

ax.set_xlabel('Source')
ax.set_ylabel('Percentage of Sentences Labeled Objective')
ax.set_title('Opinion Editorials vs. All Articles')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(sources)
ax.legend()

fig.tight_layout()
plt.show()
