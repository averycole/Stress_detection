import tensorflow as tf
import numpy as py
import pickle
import wfdb

train_labels = []
for s in range(20):
	ATE = wfdb.rdrecord('Dataset/Subject'+str(s+1)+'_AccTempEDA')
	ann = wfdb.rdann('Dataset/Subject'+str(s+1)+'_AccTempEDA','atr')
	train_labels.clear()
	annCounter = 0;
	for i in range(ATE.sig_len):
		if annCounter != 7 and i>=ann.sample[annCounter+1]:
			annCounter+=1
		if annCounter == 0 or annCounter == 2 or annCounter == 5 or annCounter == 7:
			train_labels.append(0)	#rest
		elif	annCounter == 1:
			train_labels.append(1) #pstress
		elif annCounter == 3:
			train_labels.append(2) #mstress
		elif annCounter == 4:
			train_labels.append(3) #cstress
		elif annCounter == 6:
			train_labels.append(4) #estress
	with open('Dataset/Subject'+str(s+1)+'_labels.p','wb') as fp:
		pickle.dump(train_labels,fp)
