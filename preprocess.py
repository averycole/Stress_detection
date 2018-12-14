import numpy as py
import pickle
import scipy.signal
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler

for s in range(20):
	with open('Noisy_Dataset/Sample'+str(s+1)+'.p','rb') as fp:
		raw_data = py.array(pickle.load(fp))
	hr = raw_data[:,4]
	for i in range(len(hr)):
		if hr[i]==0 and i != 0:
			print("zerod")
			hr[i]=hr[i-1]
	b, a = scipy.signal.butter(2, (110/60)/8, analog=False)
	hr_filtered = scipy.signal.filtfilt(b, a, hr)
	raw_data[:,4] = hr_filtered
	scaler=StandardScaler()
	scaler.fit(raw_data)
	data=scaler.transform(raw_data)
	with open('Preprocessed_Dataset/Sample'+str(s+1)+'.p', 'wb') as fp:
		pickle.dump(data, fp)
			
			
