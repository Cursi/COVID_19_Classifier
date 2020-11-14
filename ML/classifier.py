import re
import io
import sys
import math
import base64
import pickle
import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

from datetime import datetime
from dateutil.parser import parse

df = None

istoric = ["daistoric", "nuistoric", "missingistoric"]
contact = ["dacontact", "nucontact", "nustiecontact", "missingcontact"]
transport = ["datransport", "nutransport", "missingtransport"]

matches_contact_nustie = ["stie", "știe", "cunoaste"]
matches_contact_nu = ["fara", "nu", "0", "1", "neagă", "neaga", "nascut"]
matches_transport_nu = ["cazul", "nu", "nui", "neaga", "fara"]

simptome_raportate = ["febr1", "asim1", "tuse1", "disp1", "mialg1", "fris1", "cefal1", "odin1",
"aste1n", "fatiga1", "sub1", "cianoz1", "inapetent1", "great1", "grava1", "anosmi1", "tegumente1","abdo1",
"torac1", "muscula1", "hta1"]

diagnostic = ["bronho2", "susp2", "tuse2", "odino2", "fris2", "cefal2", "febr2", "hta2", "mialg2", "disp2", 
"gang2", "insuf2", "infec2", "pneumonie2", "respiratorie2" ]  #la insuficienta si respiratorie or sa puna doi de 1 desi e acelasi item

simptome_declarate = ["febra3", "tuse3", "dispn3", "asim3", "mialg3", "asten3", "cefalee3", "inapetent3", 
"subfe3", "fris3", "disfag3", "fatig3", "greuturi3", "greata3", "muscu3", "toracic3" ] #avem si greturi si greata aia e

df_upsampled = None
featuresDF = None

X_train = None
X_test = None

Y_train = None
Y_test = None

classifier = None

def is_date(string, fuzzy=False):
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except:
      return False

def parse_date(string, tip):
	k = 0

	for i in df.index:
		if pd.isnull(df[string][i]) == False:
			if is_date(str(df[string][i])):
				df[string][i] = parse(str(df[string][i]))
				if str(df[string][i].year) == "2020":
					if tip == "debut":
						df["luna debut"][i] = df[string][i].month
						df["zi debut"][i] = df[string][i].day
						df["zi din sapt debut"][i] = df[string][i].weekday()
					if tip == "internare":
						df["luna internare"][i] = df[string][i].month
						df["zi internare"][i] = df[string][i].day
						df["zi din sapt internare"][i] = df[string][i].weekday()
					if tip == "rezultat":
						df["luna rezultat"][i] = df[string][i].month
						df["zi rezultat"][i] = df[string][i].day
						df["zi din sapt rezultat"][i] = df[string][i].weekday()
				else:
					df[string][i] = "0-0-0 00:00:00"
					if tip == "debut":
						df["luna debut"][i] = "0"
						df["zi debut"][i] = "0"
						df["zi din sapt debut"][i] = "0"
					if tip == "internare":
						df["luna internare"][i] = "0"
						df["zi internare"][i] = "0"
						df["zi din sapt internare"][i] = "0"
					if tip == "rezultat":
						df["luna rezultat"][i] = "0"
						df["zi rezultat"][i] = "0"
						df["zi din sapt rezultat"][i] = "0"
			else:
				df[string][i] = "0-0-0 00:00:00"
				if tip == "debut":
					df["luna debut"][i] = "0"
					df["zi debut"][i] = "0"
					df["zi din sapt debut"][i] = "0"
				if tip == "internare":
					df["luna internare"][i] = "0"
					df["zi internare"][i] = "0"
					df["zi din sapt internare"][i] = "0"
				if tip == "rezultat":
					df["luna rezultat"][i] = "0"
					df["zi rezultat"][i] = "0"
					df["zi din sapt rezultat"][i] = "0"
		else:
			df[string][i] = "0-0-0 00:00:00"
			if tip == "debut":
				df["luna debut"][i] = "0"
				df["zi debut"][i] = "0"
				df["zi din sapt debut"][i] = "0"
			if tip == "internare":
				df["luna internare"][i] = "0"
				df["zi internare"][i] = "0"
				df["zi din sapt internare"][i] = "0"
			if tip == "rezultat":
				df["luna rezultat"][i] = "0"
				df["zi rezultat"][i] = "0"
				df["zi din sapt rezultat"][i] = "0"

def SetPandasCustomizations():
	pd.options.mode.chained_assignment = None

def ReadBase64Excel():
	global df

	try:
		base64_dataset = input()
		decrypted_dataset = base64.b64decode(base64_dataset)

		memoryBuffer = io.BytesIO()
		memoryBuffer.write(decrypted_dataset)
		memoryBuffer.seek(0)
		
		df = pd.read_excel(memoryBuffer)
		# df = pd.read_excel(sys.argv[1])
	except:
		print("PROCESSING_ERROR")
		exit()
	
def EncodeXColumns():
	global df

	df["instituția sursă"] = df["instituția sursă"].str.lower()
	df["instituția sursă"] = df["instituția sursă"].str.strip()
	df["instituția sursă"] = df["instituția sursă"].replace({ "x": 1, "y": 2, "z": 3 })
	df = df.dropna(subset=['instituția sursă'])
	df["inst_1"] = df["inst_2"] = df["inst_3"] = 0
	df.loc[df["instituția sursă"] == 1, 'inst_1'] = 1
	df.loc[df["instituția sursă"] == 2, 'inst_2'] = 1
	df.loc[df["instituția sursă"] == 3, 'inst_3'] = 1

	df["sex"] = df["sex"].str.lower()
	df["sex"] = df["sex"].str.strip()
	df["sex"] = df["sex"].replace({ "masculin": 1, "feminin": 2, "f": 2 })
	df = df.dropna(subset=['sex'])
	df["sex_1"] = df["sex_2"] = 0
	df.loc[df["sex"] == 1, 'sex_1'] = 1
	df.loc[df["sex"] == 2, 'sex_2'] = 1

	df["vârstă"] = pd.to_numeric(df["vârstă"], errors="coerce")
	df = df.dropna(subset=['vârstă'])

	df["age_1"] = df["age_2"] = df["age_3"] = df["age_4"] = 0

	df.loc[df['vârstă'] < 18, 'vârstă'] = 1
	df.loc[(df['vârstă'] >= 18) & (df['vârstă'] < 35), 'vârstă'] = 2
	df.loc[(df['vârstă'] >= 35) & (df['vârstă'] < 55), 'vârstă'] = 3
	df.loc[df['vârstă'] >= 55, 'vârstă'] = 4

	df["istoric de călătorie"] = df["istoric de călătorie"].str.lower()
	df["istoric de călătorie"] = df["istoric de călătorie"].str.strip()
	df["istoric de călătorie"].replace(np.nan, "-", inplace=True)

	df["mijloace de transport folosite"] = df["mijloace de transport folosite"].str.lower()
	df["mijloace de transport folosite"].replace(np.nan, "-", inplace=True)

	df["confirmare contact cu o persoană infectată"] = df["confirmare contact cu o persoană infectată"].str.lower()
	df["confirmare contact cu o persoană infectată"] = df["confirmare contact cu o persoană infectată"].str.strip()
	df["confirmare contact cu o persoană infectată"].replace(np.nan, "-", inplace=True)

	df["simptome raportate la internare"] = df["simptome raportate la internare"].str.lower()
	df["simptome raportate la internare"] = df["simptome raportate la internare"].str.strip()
	df["simptome raportate la internare"].replace(np.nan, "-", inplace=True)

	df["diagnostic și semne de internare"] = df["diagnostic și semne de internare"].str.lower()
	df["diagnostic și semne de internare"] = df["diagnostic și semne de internare"].str.strip()
	df["diagnostic și semne de internare"].replace(np.nan, "-", inplace=True)

	df["simptome declarate"] = df["simptome declarate"].str.lower()
	df["simptome declarate"] = df["simptome declarate"].str.strip()
	df["simptome declarate"].replace(np.nan, "-", inplace=True)

	for x in istoric:
		df[x] = 0

	for x in contact:
		df[x] = 0

	for x in transport:
		df[x] = 0

	for x in simptome_raportate:
		df[x] = 0

	for x in diagnostic:
		df[x] = 0

	for x in simptome_declarate:
		df[x] = 0

	for ind in df.index:
		if "nu" in df["istoric de călătorie"][ind]:
			df["nuistoric"][ind] = 1
		elif "mu" in df["istoric de călătorie"][ind]:
			df["nuistoric"][ind] = 1
		elif "far" in df["istoric de călătorie"][ind]:
			df["nuistoric"][ind] = 1
		elif "neaga" in df["istoric de călătorie"][ind]:
			df["nuistoric"][ind] = 1
		elif "neagă" in df["istoric de călătorie"][ind]:
			df["nuistoric"][ind] = 1
		elif "-" == df["istoric de călătorie"][ind]:
			df["missingistoric"][ind] = 1
		else:
			df["daistoric"][ind] = 1

		for j in simptome_raportate:
			x = j[0:(len(j)-1)]
			if x in df["simptome raportate la internare"][ind]:
				df[j][ind] = 1

		for j in diagnostic:
			x = j[0:(len(j)-1)]
			if x in df["diagnostic și semne de internare"][ind]:
				df[j][ind] = 1

		for j in simptome_declarate:
			x = j[0:(len(j)-1)]
			if x in df["simptome declarate"][ind]:
				df[j][ind] = 1

		##pe aiai cu nascuti i-am trecut la nu, poate ar trebui da
		#un singur tip cu 1 care e negativ ca rezultat
		if "-" == df["confirmare contact cu o persoană infectată"][ind]:
			df["missingcontact"][ind] = 1
		elif any(x in df["confirmare contact cu o persoană infectată"][ind] for x in matches_contact_nustie):
			df["nustiecontact"][ind] = 1
		elif any(x in df["confirmare contact cu o persoană infectată"][ind] for x in matches_contact_nu): 
			df["nucontact"][ind] = 1                                                                    

		else: #########zicem ca posibil inseamna da, "da" inseamna da, desi pareau negativi
			df["dacontact"][ind] = 1

		if "-" == df["mijloace de transport folosite"][ind]:
			df["missingtransport"][ind] = 1
		elif any(x in df["mijloace de transport folosite"][ind] for x in matches_transport_nu):
			df["nutransport"][ind] = 1                                                             
		else: 
			df["datransport"][ind] = 1

	for i in df.index:
		if ',' in str(df["dată debut simptome declarate"][i]):
			df["dată debut simptome declarate"][i] = str(df["dată debut simptome declarate"][i]).replace(',', '-')
		if '.' in str(df["dată debut simptome declarate"][i]):
			df["dată debut simptome declarate"][i] = str(df["dată debut simptome declarate"][i]).replace('.', '-')

	df["zi debut"] = df["luna debut"] = df["zi din sapt debut"] = df["zi internare"] = df["luna internare"] = df["zi din sapt internare"] = df["zi rezultat"] = df["luna rezultat"] = df["zi din sapt rezultat"] = 0

	parse_date("dată debut simptome declarate", "debut")
	parse_date("dată internare", "internare")
	parse_date("data rezultat testare", "rezultat")

def EncodeYColumns():
	global df

	df["rezultat testare"] = df["rezultat testare"].str.lower()
	df["rezultat testare"] = df["rezultat testare"].str.strip()
	df["rezultat testare"] = df["rezultat testare"].replace({"neconcludent": np.nan, "negatib": "negativ"})
	df["rezultat testare"] = df["rezultat testare"].replace({ "negativ": 0, "pozitiv": 1 })

	df = df.dropna(subset=['rezultat testare'])

def ForceConvertObjectsToIntegers():
	global df

	df["instituția sursă"] = df["instituția sursă"].astype("int64")
	df["sex"] = df["sex"].astype("int64")
	df["vârstă"] = df["vârstă"].astype("int64")
	df["rezultat testare"] = df["rezultat testare"].astype("int64")

def ComputeUpsampledDataFrame():
	global df_upsampled

	dfNegativ = df[df["rezultat testare"] == 0]
	dfPozitiv = df[df["rezultat testare"] == 1]

	dfPozitiv_Upsampled = resample(dfPozitiv, replace=True, n_samples=5069, random_state=4)
	df_upsampled = pd.concat([dfNegativ, dfPozitiv_Upsampled])

def SelectFeatures():
	global featuresDF

	featuresDF = [
		"instituția sursă",
		"sex",
		"vârstă",

		"daistoric", 
		"nuistoric",

		"dacontact", 
		"nucontact",

		"datransport", 
		"nutransport",

		"zi debut", 
		"luna debut",
		"zi din sapt debut",

		"zi internare",
		"luna internare",
		"zi din sapt internare",
		
		"zi rezultat", 
		"luna rezultat",
		"zi din sapt rezultat",

		"febr1", "asim1", "tuse1", "disp1", "mialg1", "fris1", "cefal1", "odin1",
		"aste1n", "fatiga1", "sub1", "cianoz1", "inapetent1", "great1", "grava1", "anosmi1", "tegumente1","abdo1",
		"torac1", "muscula1", "hta1",

		"bronho2", "susp2", "tuse2", "odino2", "fris2", "cefal2", "febr2", "hta2", "mialg2", "disp2", 
		"gang2", "insuf2", "infec2", "pneumonie2", "respiratorie2",

		"febra3", "tuse3", "dispn3", "asim3", "mialg3", "asten3", "cefalee3", "inapetent3", 
		"subfe3", "fris3", "disfag3", "fatig3", "greuturi3", "greata3", "muscu3", "toracic3"
]

# TODO: De facut dupa un argument fie spargerea in X Y de test si train, fie incarcarea unui train dataset
def SplitDataframe():
	global X_train, X_test, Y_train, Y_test, df_upsampled

	X = np.asarray(df_upsampled[featuresDF]) # Pe astea vreau sa le folosesc sa fac predictia clasei (cancerigen sau nu) - variabile INDEPENDENTE
	Y = np.asarray(df_upsampled['rezultat testare']) # Aci am rezultatul real cu care voi putea sa compar - variabile DEPENDENTE

	(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=4)

	# X = np.asarray(pd.read_excel("X.xlsx"))
	# Y = np.asarray(pd.read_excel("Y.xlsx"))
	# (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=4)

def LoadModel():
	global classifier

	modelFile = open('ML\classifier.model','rb')
	classifier = pickle.load(modelFile)
	modelFile.close()

def PrintPredictionMetrics():
	Y_predicted = classifier.predict(X_test)
	print(classification_report(Y_test, Y_predicted))
	# print("Confusion matrix:")
	# print(confusion_matrix(Y_test, Y_predicted))
	# print()
	# print("AUCROC score:")
	# print(roc_auc_score(Y_test, Y_predicted))

if __name__ == "__main__":
	SetPandasCustomizations()
	ReadBase64Excel()
	EncodeXColumns()
	EncodeYColumns()
	ForceConvertObjectsToIntegers()
	ComputeUpsampledDataFrame()
	SelectFeatures()
	SplitDataframe()
	LoadModel()
	PrintPredictionMetrics()