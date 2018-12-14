import numpy as py
import wfdb
import sklearn as sk
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier
import csv
import pickle

SVM_score = []
SVM_prob = []
DT_score = []
DT_prob = []
MLP_score = []
MLP_prob = []
for t in range(20):
	train_data = []
	train_labels = []
	for s in range(20):
		if s!=t:
			with open('Preprocessed_Dataset/Sample'+str(s+1)+'.p','rb') as fp:
				data_set = pickle.load(fp)
			train_data.extend(data_set)
	
			with open('Dataset/Subject'+str(s+1)+'_labels.p','rb') as fp:
				label_set = pickle.load(fp)
			train_labels.extend(label_set)
	SVMc = svm.SVC(gamma='scale',probability=True)
	SVMc = SVMc.fit(train_data, train_labels)
	DT = RandomForestClassifier(n_estimators=2)
	DT = DT.fit(train_data, train_labels)
	MLP = MLPClassifier(batch_size=1000, max_iter=5000)
	MLP = MLP.fit(train_data, train_labels)
	
	with open('Preprocessed_Dataset/Sample'+str(t+1)+'.p','rb') as fp:
		test_data = pickle.load(fp)
	with open('Dataset/Subject'+str(t+1)+'_labels.p','rb') as fp:
		test_labels = pickle.load(fp)
	
	SVMs = SVMc.score(test_data,test_labels)
	SVM_score.append(SVMs)
	DTs = DT.score(test_data,test_labels)
	DT_score.append(DTs)
	MLPs = MLP.score(test_data,test_labels)
	MLP_score.append(MLPs)
	print('trial '+str(t))
	if t == 0:
		SVM_prob=SVMc.predict_proba(test_data)
		DT_prob=DT.predict_proba(test_data)
		MLP_prob=MLP.predict_proba(test_data)
		SVM_p = open('Outputs/SVM_probs.csv','w')
		DT_p = open('Outputs/DT_probs.csv','w')
		MLP_p = open('Outputs/MLP_probs.csv','w')
		with SVM_p:	
			writer = csv.writer(SVM_p)
			writer.writerows(SVM_prob)
		with DT_p:	
			writer = csv.writer(DT_p)
			writer.writerows(DT_prob)
		with MLP_p:	
			writer = csv.writer(MLP_p)
			writer.writerows(MLP_prob)
	SVM_s = open('Outputs/SVM_scores.csv','w')
	DT_s = open('Outputs/DT_scores.csv','w')
	MLP_s = open('Outputs/MLP_scores.csv','w')
	with SVM_s:	
		writer = csv.writer(SVM_s)
		writer.writerow(SVM_score)
	with DT_s:	
		writer = csv.writer(DT_s)
		writer.writerow(DT_score)
	with MLP_s:	
		writer = csv.writer(MLP_s)
		writer.writerow(MLP_score)
