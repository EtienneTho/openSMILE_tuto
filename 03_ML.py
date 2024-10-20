import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pingouin as pg
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import RobustScaler, StandardScaler
from lib import proise_v2


data = pickle.load(open('./features_OpenSMILE.pkl', 'rb'))

X = data['dataOpenSMILE']
X[:,235] = np.random.randn(X.shape[0],)
X[:,323] = np.random.randn(X.shape[0],)

scaler = StandardScaler()
X = scaler.fit_transform(X)

class_ = 'species'  # species or ids
y = data[class_]

n_fold = 10
n_components = 20
n_cv = 2
cv = StratifiedKFold(n_cv, shuffle=True)

explained_varianceTab = []
testScore = []
canonicalAllMaps = []
canonicalAllMaps_pval = []

for fold in range(n_fold):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=fold)
  
  pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train)
  X_train_pca = pca.transform(np.asarray(X_train))
  # print('explained variance :'+str(np.sum(pca.explained_variance_ratio_)))
  explained_varianceTab.append(np.sum(pca.explained_variance_ratio_))
  # print('PCA done '+str(n_components))

  # grid search classifier definition and fit
  # tuned_parameters = { 'gamma':np.logspace(-3,3,num=3),'C':np.logspace(-3,3,num=3)}
  # clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, n_jobs=-1, cv=cv,  pre_dispatch=6,
  #                scoring='balanced_accuracy', verbose=False)
  clf = SVC(kernel='rbf')
  clf.fit(X_train_pca, y_train)
  # print('GridSearch done')

  y_pred = clf.predict(pca.transform(X_test))
  testScore.append(balanced_accuracy_score(y_test,y_pred))

  ################################################################################################################################
  # interpretation stim+pseudo-noise
  N = 20000 # for bubbles
  N = 4000 # for revcor
  dimOfinput = (1,530) # dimension of the input representation

  # additive noise
  probingMethod = 'revcor' # choice of the probing method : bubbles or revcor
  samplesMethod = 'pseudoRandom' # choice of the method to generate probing samples trainSet or pseudoRandom (need x_test_set) or gaussianNoise
  nbRevcorTrials = X_train.shape[0]*N # number of probing samples for the reverse correlation (must be below the number of training sample if trainSet)
  nDim_pca = n_components # number of dimension to compute the PCA for the pseudo-random noise generation
  probingSamples, _ = proise_v2.generateProbingSamples(x_train_set = X_train, x_test_set = X_train, dimOfinput=dimOfinput, probingMethod = probingMethod, samplesMethod = samplesMethod, nDim_pca = nDim_pca, nbRevcorTrials = nbRevcorTrials)
  X_probingSamples_pca = pca.transform(probingSamples/2 + np.tile(X_train,(N,1))/2)  
  data2revcor = np.asarray(probingSamples)
  print(probingSamples.shape)
  # Initialize an array of zeros

  # # bubbles
  # nb_bubbles = 200
  # bubbleMask = np.zeros((X_train.shape[0]*N, data['colNames'].shape[0]), dtype=int)

  # # Loop through each row and assign exactly N ones randomly
  # for i in range(X_train.shape[0]*N):
  #     row = np.zeros(data['colNames'].shape[0], dtype=int)  # Start with a row of zeros
  #     row[:nb_bubbles] = 1                   # Set the first N elements to 1
  #     np.random.shuffle(row)        # Shuffle the row to randomly distribute ones
  #     bubbleMask[i] = row               # Assign the row to the matrix
  # X_probingSamples_pca = pca.transform(np.tile(X_train,(N,1))/2 * bubbleMask)  
  # data2revcor = np.asarray(bubbleMask)

  y_pred = clf.predict(X_probingSamples_pca)
  responses_ = np.asarray(np.squeeze((y_pred == np.tile(y_train,(1,N)))))

  canonicalMap = []
  pval = []

  
  

  # canonicalMap = np.nanmean(data2revcor[responses_][:],axis=0) - np.nanmean(data2revcor[np.logical_not(responses_)][:],axis=0)


  for iFeature in range(data['colNames'].shape[0]):
    if np.unique(data2revcor[:,iFeature]).shape[0] == 1:
      data2revcor[:,iFeature] = np.random.randn(X_train.shape[0]*N,)
    corr_ = pg.corr(data2revcor[:,iFeature],responses_).round(3)
    canonicalMap.append(corr_['r'].iloc[0])
    pval.append(corr_['p-val'].iloc[0])
    # canonicalMap=0  
  
  canonicalAllMaps.append(np.asarray(canonicalMap))
  canonicalAllMaps_pval.append(np.asarray(pval))
print('Averaged explained variance: M='+str(np.nanmean(explained_varianceTab))+', SD='+str(np.nanstd(explained_varianceTab)))
print('Averaged test score: M='+str(np.mean(testScore))+', SD='+str(np.std(testScore)))

plt.plot(np.nanmean(canonicalAllMaps_pval,axis=0))
plt.show()

mean_canonical_map = np.nanmean(canonicalAllMaps,axis=0)
plt.plot(mean_canonical_map)
pval_threshold = .05
clean_canonical_map = mean_canonical_map
clean_canonical_map[np.nanmean(canonicalAllMaps_pval,axis=0)>pval_threshold] = np.nan

plt.scatter(range(data['colNames'].shape[0]),clean_canonical_map,c='r')
# plt.axis([0, 531, -.001, .03])
plt.axis([0, 531, -.1, .1])
plt.xlabel('openSMILE features')
plt.ylabel('reverse correlation weights')
plt.savefig('./out_figures/00000'+class_+'canonical_map.pdf', format='pdf')
plt.show()

print(data['colNames'][np.nanmean(canonicalAllMaps_pval,axis=0)<pval_threshold])



