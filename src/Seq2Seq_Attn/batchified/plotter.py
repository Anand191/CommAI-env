import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set()
fpath = './Infer_Results/Model'

df = 
f = plt.figure()
ax = f.add_subplot(111)
plt.title('Overall', fontsize=20)
df.plot(kind='bar', ax=f.gca(), rot=0, fontsize=11)
# lgd = ax.legend(loc='center', bbox_to_anchor=(0.5,-0.35), ncol=2, fontsize=12)
# f.savefig('Longer.png', bbox_extra_artists=(lgd,), bbox_inches='tight')