import re
import os
import io
import sys
import math
import json
import base64
import pickle
import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, f1_score, recall_score

from datetime import datetime
from dateutil.parser import parse

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

fileName = None
df = None

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
	global df, fileName

	if len(sys.argv) == 3:
		try:
			if sys.argv[1] == "1":
				base64_dataset = input()
				decrypted_dataset = base64.b64decode(base64_dataset)

				memoryBuffer = io.BytesIO()
				memoryBuffer.write(decrypted_dataset)
				memoryBuffer.seek(0)
				
				fileName = sys.argv[2]
				df = pd.read_excel(memoryBuffer)
			elif sys.argv[1] == "0":
				df = pd.read_excel(sys.argv[2])
		except:
			print("PROCESSING_ERROR")
			exit()
	
def EncodeXColumns():
	global df

	df["instituția sursă"] = df["instituția sursă"].str.lower()
	df["instituția sursă"] = df["instituția sursă"].str.strip()
	df["instituția sursă"] = df["instituția sursă"].replace({ "x": 1, "y": 2, "z": 3 })
	df['instituția sursă'] = df['instituția sursă'].fillna(0)

	df["sex"] = df["sex"].str.lower()
	df["sex"] = df["sex"].str.strip()
	df["sex"] = df["sex"].replace({ "masculin": 1, "feminin": 2, "f": 2 })
	df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
	df['sex'] = df['sex'].fillna(0)

	df["vârstă"] = pd.to_numeric(df["vârstă"], errors="coerce")
	df['vârstă'] = df['vârstă'].fillna(0)

	df["simptome raportate la internare"] = df["simptome raportate la internare"].str.lower()
	df["simptome raportate la internare"].replace(np.nan, "-", inplace=True)

	df["diagnostic și semne de internare"] = df["diagnostic și semne de internare"].str.lower()
	df["diagnostic și semne de internare"].replace(np.nan, "-", inplace=True)

	df["simptome declarate"] = df["simptome declarate"].str.lower()
	df["simptome declarate"].replace(np.nan, "-", inplace=True)

	for x in simptome_raportate:
		df[x] = 0

	for x in diagnostic:
		df[x] = 0

	for x in simptome_declarate:
		df[x] = 0

	for ind in df.index:
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
	df["rezultat testare"] = pd.to_numeric(df["rezultat testare"], errors="coerce")
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

	dfPozitiv_Upsampled = resample(dfPozitiv, replace=True, n_samples=dfNegativ.shape[0], random_state=4)
	df_upsampled = pd.concat([dfNegativ, dfPozitiv_Upsampled])

def SelectFeatures():
	global featuresDF

	featuresDF = [
		"instituția sursă",
		"sex",
		"vârstă",

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

def SplitDataframe():
	global X_train, X_test, Y_train, Y_test, df_upsampled

	if sys.argv[1] == "0":
		X = np.asarray(df_upsampled[featuresDF])
		Y = np.asarray(df_upsampled['rezultat testare'])

		(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.1, random_state=4)
	elif sys.argv[1] == "1":
		X_test = np.asarray(df[featuresDF])
		Y_test = np.asarray(df['rezultat testare'])

def LoadModel():
	global classifier

	modelFile = open('adaBoostedRFC_10Estimators.model','rb')
	classifier = pickle.load(modelFile)
	modelFile.close()

def PrintPredictionMetrics():
	Y_predicted = classifier.predict(X_test)
	
	if sys.argv[1] == "0":
		print(classification_report(Y_test, Y_predicted))
		print("Confusion matrix:")
		print(confusion_matrix(Y_test, Y_predicted))
		print()
		print("AUCROC score:")
		print(roc_auc_score(Y_test, Y_predicted))
	elif sys.argv[1] == "1":
		jsonObject = {
			"file_name": sys.argv[2],
			"accuracy": accuracy_score(Y_test, Y_predicted),
			"aucroc": roc_auc_score(Y_test, Y_predicted),
			"precision": precision_score(Y_test, Y_predicted, average=None),
			"recall": recall_score(Y_test, Y_predicted, average=None),
			"f1": f1_score(Y_test, Y_predicted, average=None),
			"confusion_matrix": confusion_matrix(Y_test, Y_predicted)
		}

		jsonString = json.dumps(jsonObject, cls=NumpyEncoder)
		print(jsonString)
	
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