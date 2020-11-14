import sys
import pandas as pd
import numpy as np
import math
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from dateutil.parser import parse
from datetime import datetime

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import svm

from sklearn.utils import resample

# def is_date(string, fuzzy=False):
#     try: 
#         parse(string, fuzzy=fuzzy)
#         return True

#     except:
#       return False

# def parse_date(string, tip):
# 	k = 0
# 	for i in df.index:
# 		if pd.isnull(df[string][i]) == False:
# 			if is_date(str(df[string][i])):
# 				df[string][i] = parse(str(df[string][i]))
# 				if str(df[string][i].year) == "2020": #avem 2 cazuri de 020 la an
# 					if tip == "debut":
# 						df["luna debut"][i] = df[string][i].month
# 						df["zi debut"][i] = df[string][i].day
# 						df["zi din sapt debut"][i] = df[string][i].weekday()
# 					if tip == "internare":
# 						df["luna internare"][i] = df[string][i].month
# 						df["zi internare"][i] = df[string][i].day
# 						df["zi din sapt internare"][i] = df[string][i].weekday()
# 					if tip == "rezultat":
# 						df["luna rezultat"][i] = df[string][i].month
# 						df["zi rezultat"][i] = df[string][i].day
# 						df["zi din sapt rezultat"][i] = df[string][i].weekday()
# 				else:
# 					df[string][i] = "0-0-0 00:00:00"
# 					if tip == "debut":
# 						df["luna debut"][i] = "0"
# 						df["zi debut"][i] = "0"
# 						df["zi din sapt debut"][i] = "0"
# 					if tip == "internare":
# 						df["luna internare"][i] = "0"
# 						df["zi internare"][i] = "0"
# 						df["zi din sapt internare"][i] = "0"
# 					if tip == "rezultat":
# 						df["luna rezultat"][i] = "0"
# 						df["zi rezultat"][i] = "0"
# 						df["zi din sapt rezultat"][i] = "0"
# 			else:
# 				df[string][i] = "0-0-0 00:00:00"
# 				if tip == "debut":
# 					df["luna debut"][i] = "0"
# 					df["zi debut"][i] = "0"
# 					df["zi din sapt debut"][i] = "0"
# 				if tip == "internare":
# 					df["luna internare"][i] = "0"
# 					df["zi internare"][i] = "0"
# 					df["zi din sapt internare"][i] = "0"
# 				if tip == "rezultat":
# 					df["luna rezultat"][i] = "0"
# 					df["zi rezultat"][i] = "0"
# 					df["zi din sapt rezultat"][i] = "0"
# 		else:
# 			df[string][i] = "0-0-0 00:00:00"
# 			if tip == "debut":
# 				df["luna debut"][i] = "0"
# 				df["zi debut"][i] = "0"
# 				df["zi din sapt debut"][i] = "0"
# 			if tip == "internare":
# 				df["luna internare"][i] = "0"
# 				df["zi internare"][i] = "0"
# 				df["zi din sapt internare"][i] = "0"
# 			if tip == "rezultat":
# 				df["luna rezultat"][i] = "0"
# 				df["zi rezultat"][i] = "0"
# 				df["zi din sapt rezultat"][i] = "0"	

#### Pandas settings
pd.options.mode.chained_assignment = None  # default='warn'
# pd.set_option('display.max_rows', 1000)
# np.set_printoptions(threshold=sys.maxsize)

#### Citesc datele
# df = pd.read_excel(sys.argv[1])

import io
import base64

try:
	base64_dataset = input()
	decrypted_dataset = base64.b64decode(base64_dataset)
	toread = io.BytesIO()
	toread.write(decrypted_dataset)  # pass your `decrypted` string as the argument here
	toread.seek(0)  # reset the pointer
	# df = pd.read_excel(toread)  # now read to dataframe
except:
	print("PROCESSING_ERROR")
	exit()

# # print(df.shape)
# # print(df.isna().sum())
# # print()

# #### Imi aleg coloanele
# # df = df[['instituția sursă', 'sex', 'vârstă', 'rezultat testare', "istoric de călătorie"]]
# # print(df.shape)
# # print(df.dtypes)
# # print(df.isna().sum())
# # print(df.head())

# #### Editez coloanele
# df["instituția sursă"] = df["instituția sursă"].str.lower()
# df["instituția sursă"] = df["instituția sursă"].str.strip()
# df["instituția sursă"] = df["instituția sursă"].replace({ "x": 1, "y": 2, "z": 3 })
# df = df.dropna(subset=['instituția sursă'])
# df["inst_1"] = df["inst_2"] = df["inst_3"] = 0
# df.loc[df["instituția sursă"] == 1, 'inst_1'] = 1
# df.loc[df["instituția sursă"] == 2, 'inst_2'] = 1
# df.loc[df["instituția sursă"] == 3, 'inst_3'] = 1

# df["sex"] = df["sex"].str.lower()
# df["sex"] = df["sex"].str.strip()
# df["sex"] = df["sex"].replace({ "masculin": 1, "feminin": 2, "f": 2 })
# df = df.dropna(subset=['sex'])
# df["sex_1"] = df["sex_2"] = 0
# df.loc[df["sex"] == 1, 'sex_1'] = 1
# df.loc[df["sex"] == 2, 'sex_2'] = 1

# df["vârstă"] = pd.to_numeric(df["vârstă"], errors="coerce")
# df = df.dropna(subset=['vârstă'])

# df["age_1"] = df["age_2"] = df["age_3"] = df["age_4"] = 0
# # df.loc[df['vârstă'] < 18, 'age_1'] = 1
# # df.loc[(df['vârstă'] >= 18) & (df['vârstă'] < 35), 'age_2'] = 1
# # df.loc[(df['vârstă'] >= 35) & (df['vârstă'] < 55), 'age_3'] = 1
# # df.loc[df['vârstă'] >= 55, 'age_4'] = 1

# df.loc[df['vârstă'] < 18, 'vârstă'] = 1
# df.loc[(df['vârstă'] >= 18) & (df['vârstă'] < 35), 'vârstă'] = 2
# df.loc[(df['vârstă'] >= 35) & (df['vârstă'] < 55), 'vârstă'] = 3
# df.loc[df['vârstă'] >= 55, 'vârstă'] = 4

# istoric = ["daistoric", "nuistoric", "missingistoric"]
# contact = ["dacontact", "nucontact", "nustiecontact", "missingcontact"]
# transport = ["datransport", "nutransport", "missingtransport"]

# matches_contact_nustie = ["stie", "știe", "cunoaste"]
# matches_contact_nu = ["fara", "nu", "0", "1", "neagă", "neaga", "nascut"]
# matches_transport_nu = ["cazul", "nu", "nui", "neaga", "fara"]

# simptome_raportate = ["febr1", "asim1", "tuse1", "disp1", "mialg1", "fris1", "cefal1", "odin1",
# "aste1n", "fatiga1", "sub1", "cianoz1", "inapetent1", "great1", "grava1", "anosmi1", "tegumente1","abdo1",
#  "torac1", "muscula1", "hta1"]

# diagnostic = ["bronho2", "susp2", "tuse2", "odino2", "fris2", "cefal2", "febr2", "hta2", "mialg2", "disp2", 
# "gang2", "insuf2", "infec2", "pneumonie2", "respiratorie2" ]  #la insuficienta si respiratorie or sa puna doi de 1 desi e acelasi item

# simptome_declarate = ["febra3", "tuse3", "dispn3", "asim3", "mialg3", "asten3", "cefalee3", "inapetent3", 
# "subfe3", "fris3", "disfag3", "fatig3", "greuturi3", "greata3", "muscu3", "toracic3" ] #avem si greturi si greata aia e

# df["istoric de călătorie"] = df["istoric de călătorie"].str.lower()
# df["istoric de călătorie"] = df["istoric de călătorie"].str.strip()
# df["istoric de călătorie"].replace(np.nan, "-", inplace=True)

# df["mijloace de transport folosite"] = df["mijloace de transport folosite"].str.lower()
# # df["mijloace de transport folosite"] = df["mijloace de transport folosite"].str.strip()
# df["mijloace de transport folosite"].replace(np.nan, "-", inplace=True)

# df["confirmare contact cu o persoană infectată"] = df["confirmare contact cu o persoană infectată"].str.lower()
# df["confirmare contact cu o persoană infectată"] = df["confirmare contact cu o persoană infectată"].str.strip()
# df["confirmare contact cu o persoană infectată"].replace(np.nan, "-", inplace=True)

# df["simptome raportate la internare"] = df["simptome raportate la internare"].str.lower()
# df["simptome raportate la internare"] = df["simptome raportate la internare"].str.strip()
# df["simptome raportate la internare"].replace(np.nan, "-", inplace=True)

# df["diagnostic și semne de internare"] = df["diagnostic și semne de internare"].str.lower()
# df["diagnostic și semne de internare"] = df["diagnostic și semne de internare"].str.strip()
# df["diagnostic și semne de internare"].replace(np.nan, "-", inplace=True)

# df["simptome declarate"] = df["simptome declarate"].str.lower()
# df["simptome declarate"] = df["simptome declarate"].str.strip()
# df["simptome declarate"].replace(np.nan, "-", inplace=True)



# for x in istoric:
# 	df[x] = 0

# for x in contact:
# 	df[x] = 0

# for x in transport:
# 	df[x] = 0

# for x in simptome_raportate:
# 	df[x] = 0

# for x in diagnostic:
# 	df[x] = 0

# for x in simptome_declarate:
# 	df[x] = 0

# for ind in df.index:
#     if "nu" in df["istoric de călătorie"][ind]:
#     	df["nuistoric"][ind] = 1
#     elif "mu" in df["istoric de călătorie"][ind]:
#     	df["nuistoric"][ind] = 1
#     elif "far" in df["istoric de călătorie"][ind]:
#     	df["nuistoric"][ind] = 1
#     elif "neaga" in df["istoric de călătorie"][ind]:
#     	df["nuistoric"][ind] = 1
#     elif "neagă" in df["istoric de călătorie"][ind]:
#     	df["nuistoric"][ind] = 1
#     elif "-" == df["istoric de călătorie"][ind]:
#     	df["missingistoric"][ind] = 1
#     else:
#     	df["daistoric"][ind] = 1

#     for j in simptome_raportate:
# 	    x = j[0:(len(j)-1)]
# 	    if x in df["simptome raportate la internare"][ind]:
# 	    	df[j][ind] = 1

#     for j in diagnostic:
# 	    x = j[0:(len(j)-1)]
# 	    if x in df["diagnostic și semne de internare"][ind]:
# 	    	df[j][ind] = 1

#     for j in simptome_declarate:
# 	    x = j[0:(len(j)-1)]
# 	    if x in df["simptome declarate"][ind]:
# 	    	df[j][ind] = 1

#     ##pe aiai cu nascuti i-am trecut la nu, poate ar trebui da
#     #un singur tip cu 1 care e negativ ca rezultat
#     if "-" == df["confirmare contact cu o persoană infectată"][ind]:
#     	df["missingcontact"][ind] = 1
#     elif any(x in df["confirmare contact cu o persoană infectată"][ind] for x in matches_contact_nustie):
#     	df["nustiecontact"][ind] = 1
#     elif any(x in df["confirmare contact cu o persoană infectată"][ind] for x in matches_contact_nu): 
#     	df["nucontact"][ind] = 1                                                                    

#     else: #########zicem ca posibil inseamna da, "da" inseamna da, desi pareau negativi
#     	df["dacontact"][ind] = 1

#     if "-" == df["mijloace de transport folosite"][ind]:
#     	df["missingtransport"][ind] = 1
#     elif any(x in df["mijloace de transport folosite"][ind] for x in matches_transport_nu):
#     	df["nutransport"][ind] = 1                                                             
#     else: 
#     	df["datransport"][ind] = 1

# for i in df.index:
# 	if ',' in str(df["dată debut simptome declarate"][i]):
# 		df["dată debut simptome declarate"][i] = str(df["dată debut simptome declarate"][i]).replace(',', '-')
# 	if '.' in str(df["dată debut simptome declarate"][i]):
# 		df["dată debut simptome declarate"][i] = str(df["dată debut simptome declarate"][i]).replace('.', '-')

# df["zi debut"] = df["luna debut"] = df["zi din sapt debut"] = df["zi internare"] = df["luna internare"] = df["zi din sapt internare"] = df["zi rezultat"] = df["luna rezultat"] = df["zi din sapt rezultat"] = 0

# #medie("dată debut simptome declarate")
# parse_date("dată debut simptome declarate", "debut")

# #medie("dată internare")
# parse_date("dată internare", "internare")

# #medie("data rezultat testare")
# parse_date("data rezultat testare", "rezultat")

# df["rezultat testare"] = df["rezultat testare"].str.lower()
# df["rezultat testare"] = df["rezultat testare"].str.strip()
# df["rezultat testare"] = df["rezultat testare"].replace({"neconcludent": np.nan, "negatib": "negativ"})
# df["rezultat testare"] = df["rezultat testare"].replace({ "negativ": 0, "pozitiv": 1 })
# df = df.dropna(subset=['rezultat testare'])

# #### Convertesc fortat tipurile de date in int64
# df["instituția sursă"] = df["instituția sursă"].astype("int64")
# df["sex"] = df["sex"].astype("int64")
# df["vârstă"] = df["vârstă"].astype("int64")
# df["rezultat testare"] = df["rezultat testare"].astype("int64")

# # print(df.shape)
# # print(df.dtypes)
# # print(df.isna().sum())
# # print(df.head())

# dfNegativ = df[df["rezultat testare"] == 0]
# dfPozitiv = df[df["rezultat testare"] == 1]

# # print(df["rezultat testare"].value_counts())

# dfPozitiv_Upsampled = resample(dfPozitiv,
#                                  replace=True,     # sample with replacement
#                                  n_samples=5069,    # to match majority class
#                                  random_state=4) # reproducible results

# # dfNegativ_Upsampled = resample(dfNegativ,
# #                                  replace=True,     # sample with replacement
# #                                  n_samples=10000,    # to match majority class
# #                                  random_state=42) # reproducible results

# # dfNegativ_Downsampled = resample(dfNegativ, 
# #                                  replace=False,    # sample without replacement
# #                                  n_samples=611,     # to match minority class
# #                                  random_state=4) # reproducible results

# df_upsampled = pd.concat([dfNegativ, dfPozitiv_Upsampled])
# # df_upsampled = pd.concat([dfNegativ_Upsampled, dfPozitiv_Upsampled])
# # print(df_upsampled["rezultat testare"].value_counts())
# # print(df_upsampled["rezultat testare"].value_counts())

# # df_downsampled = pd.concat([dfNegativ_Downsampled, dfPozitiv])
# # print(df_downsampled["rezultat testare"].value_counts())

# #### Split in X and Y
# # print("Creating and training classifier...")

# featuresDF = [
#     "instituția sursă",
#     # # 'inst_1', 'inst_2', 'inst_3', 
#     "sex",
#     # # 'sex_1', 'sex_2', 
#     "vârstă",
#     # # 'age_1', 'age_2', 'age_3', 'age_4',

#     "daistoric", 
#     "nuistoric",
#     #  "missingistoric",

#     "dacontact", 
#     "nucontact",
#     # #  "nustiecontact",
#     #  "missingcontact",

#     "datransport", 
#     "nutransport",
#     # "missingtransport",

#     "zi debut", 
#     "luna debut",
#     "zi din sapt debut",

#     "zi internare",
#     "luna internare",
#     "zi din sapt internare",
      
#     "zi rezultat", 
#     "luna rezultat",
#     "zi din sapt rezultat",

#     "febr1", "asim1", "tuse1", "disp1", "mialg1", "fris1", "cefal1", "odin1",
#     "aste1n", "fatiga1", "sub1", "cianoz1", "inapetent1", "great1", "grava1", "anosmi1", "tegumente1","abdo1",
#     "torac1", "muscula1", "hta1",

#     "bronho2", "susp2", "tuse2", "odino2", "fris2", "cefal2", "febr2", "hta2", "mialg2", "disp2", 
#     "gang2", "insuf2", "infec2", "pneumonie2", "respiratorie2",

#     "febra3", "tuse3", "dispn3", "asim3", "mialg3", "asten3", "cefalee3", "inapetent3", 
#     "subfe3", "fris3", "disfag3", "fatig3", "greuturi3", "greata3", "muscu3", "toracic3"
# ]

# # X = np.asarray(df_downsampled[featuresDF]) # Pe astea vreau sa le folosesc sa fac predictia clasei (cancerigen sau nu) - variabile INDEPENDENTE
# # Y = np.asarray(df_downsampled['rezultat testare']) # Aci am rezultatul real cu care voi putea sa compar - variabile DEPENDENTE

# X = np.asarray(df_upsampled[featuresDF]) # Pe astea vreau sa le folosesc sa fac predictia clasei (cancerigen sau nu) - variabile INDEPENDENTE
# Y = np.asarray(df_upsampled['rezultat testare']) # Aci am rezultatul real cu care voi putea sa compar - variabile DEPENDENTE

# # X = np.asarray(df[featuresDF]) # Pe astea vreau sa le folosesc sa fac predictia clasei (cancerigen sau nu) - variabile INDEPENDENTE
# # Y = np.asarray(df['rezultat testare']) # Aci am rezultatul real cu care voi putea sa compar - variabile DEPENDENTE
# # print(X.shape)
# # print(Y.shape)

# df_upsampled[featuresDF].to_excel("X.xlsx", index=False)
# df_upsampled['rezultat testare'].to_excel("Y.xlsx", index=False)

X = np.asarray(pd.read_excel("X.xlsx"))
Y = np.asarray(pd.read_excel("Y.xlsx"))
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=4)

# classifier = LogisticRegression()
# classifier = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100, random_state=4)
# classifier = RandomForestClassifier(class_weight="balanced", max_depth=6, random_state=4)
# classifier = svm.SVC(kernel="linear", gamma="auto", C=1)
# classifier = svm.SVC(kernel="linear")
# classifier = svm.SVC(kernel="poly", degree=2, gamma="auto", C=10)
# classifier = svm.SVC(kernel="linear", class_weight="balanced")

# classifier = AdaBoostClassifier(RandomForestClassifier(),n_estimators=100, random_state=4)
# classifier = RandomForestClassifier(class_weight="balanced", random_state=0)
# classifier = RandomForestClassifier(random_state=4)
# classifier = KNeighborsClassifier()
# classifier = AdaBoostClassifier(random_state=0)
# classifier = RandomForestClassifier(random_state=0, n_estimators=10)
# classifier = MLPClassifier(max_iter=2000)
# classifier = AdaBoostClassifier(RandomForestClassifier(random_state=4, n_estimators=10, criterion="entropy"), random_state=4)
# classifier = QuadraticDiscriminantAnalysis()

# param_grid = { 
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
# }

# rfc=RandomForestClassifier(random_state=4)
# classifier = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
# classifier = RandomForestClassifier(random_state=4, n_estimators=10, criterion="entropy")
# print(classifier.best_params_)

# classifier = AdaBoostClassifier(RandomForestClassifier(random_state=4, n_estimators=10, criterion="entropy"), random_state=4)
# classifier.fit(X_train, Y_train)

# print()
print("Predicting...")
import pickle
var = open('classifier.model','rb')
classifier = pickle.load(var)
var.close()

Y_predicted = classifier.predict(X_test)
print(classification_report(Y_test, Y_predicted))
print("Confusion matrix:")
print(confusion_matrix(Y_test, Y_predicted))
print()
print("AUCROC score:")
print(roc_auc_score(Y_test, Y_predicted))

# df.to_excel("test.xlsx")