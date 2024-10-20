import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pingouin as pg



data = pickle.load(open('./features_OpenSMILE.pkl', 'rb'))

species = data['species']

# example of display
human = data['dataOpenSMILE'][species=='H'][:]
monkey = data['dataOpenSMILE'][species=='M'][:]
print(human.shape)
print(monkey.shape)

# print(data['colNames'].shape)

for iFeature in tqdm(range(data['colNames'].shape[0])):
	data2plot = [human[:,iFeature],monkey[:,iFeature]]
	ttest_result = pg.ttest(human[:,iFeature], monkey[:,iFeature])
	plt.figure(figsize=(8, 6))
	sns.violinplot(data=data2plot)

	# Add labels
	plt.xticks([0, 1], ['Human', 'Monkey'])
	plt.xlabel('Species')
	plt.title('t('+str(int(ttest_result['dof']))+')='+str(float(ttest_result['T']))+', p='+str(float(ttest_result['p-val'])))
	plt.ylabel(data['colNames'][iFeature])

	# Show the plot
	plt.savefig('./out_figures/'+str(iFeature).zfill(3)+'_'+data['colNames'][iFeature]+'.pdf', format='pdf')