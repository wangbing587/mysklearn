fig,axes=plt.subplots(2,3,dpi=100)
for i in range(len(msmses1)):
    exap1 = deepcopy(msmses1[i])
    ax = sns.scatterplot(x='Retention time',y='iRT',data=msmses1[i],
                    s=10,edgecolors='black', 
                   ax=axes[i//3,i%3])
    ax.set_title('Cell {}'.format(i+1))
plt.tight_layout()
