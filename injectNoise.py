import tensorflow as tf
import numpy as py
import pickle
import wfdb
import random

zero_HR_chance =5 

annCounter = 0;
noisy_set = []
for s in range(20):
	ATE = wfdb.rdrecord('Dataset/Subject'+str(s+1)+'_AccTempEDA')
	SOH = wfdb.rdrecord('Dataset/Subject'+str(s+1)+'_SpO2HR')
	ann = wfdb.rdann('Dataset/Subject'+str(s+1)+'_AccTempEDA','atr')
	noisy_set.clear()
	for i in range(ATE.sig_len):
		samp=py.array(ATE.p_signal[i])
		if i%8==0:
			if random.randint(1,zero_HR_chance) == 0:
				SOH_samp = [0, 0]
			else:
				SOH_samp = SOH.p_signal[int(i/8)]
		samp=py.append(samp,SOH_samp)
		samp = samp[[0,1,2,4,6]]
		noisy_set.append(samp)
	with open('Noisy_Dataset/Sample'+str(s+1)+'.p', 'wb') as fp:
		pickle.dump(noisy_set,fp)	
