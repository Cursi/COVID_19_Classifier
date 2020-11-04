import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm

from dateutil.parser import parse
from datetime import datetime

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', 1000)
np.set_printoptions(threshold=sys.maxsize)

def is_date(string, fuzzy=False):
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except:
      return False
def is_int(string):
  try:
    int(string)
    return True

  except ValueError:
    return False

def medie(column):
  s = 0
  j = 0
  for i in df.index:
    if pd.isnull(df[column][i]) == False and is_int(df[column][i]):
      s += int(df[column][i])
      j += 1
    elif pd.isnull(df[column][i]) == False:
      df[column][i] = str(df[column][i]).lower()
      x = re.search(".*lun|.*nascut|.*sap|.*zi|.*or", df[column][i])
      if x:
        df[column][i] = "0"
        j += 1
      x = re.search(".*an", df[column][i])
      if x:
        df[column][i] = df[column][i][0]
        j += 1
  s /= j
  return s

def timestamp(column):
  s = 0
  j = 0
  for i in df.index:
    if pd.isnull(df[column][i]) == False:
        now =str(df[column][i])
        try:
          df[column][i] = datetime.timestamp(parse(now))
          if (df[column][i]) >= 0:
            s += df[column][i]
            j += 1
        except:
          df[column][i] = "0"
 
  s = round(s / j)
  df[column] = df[column].replace(np.nan, s)
  df[column] = df[column].replace("0", s)
  for i in df.index:
    if df[column][i] < 0:
      df[column][i] = s



simptome_declarate = ["asim1", "febr1", "tuse1", "dispn1", "tulbura1", "fatig1",
						"convulsi1", "ameteli1", "abdo1", "muscu1", "cefalee1", "edem1", "paloare1",
						"tumefie1", "echimoza1", "greturi1", "varsatu1", "diar1", "hta1",
						"trombembolie1", "fris1", "odinofag1","subf1", "mialg1", "muscula1", "asteni1",
						 "epistaxis1", "inapetent1", "epigastri1", "erupt1", "transpir1"]
 
simptome_raportate = ["asim2", "febr2", "tuse2", "dispn2", "abd2", "fris2", "toracic2", "asten2",
						"fatig2", "subf2", "diare2", "greata2", "cefalee2", "edeme2", "varsatu2",
						"mialg2", "inapetent2", "grav2", "disfagie2", "faringe2", "epigastr2","palpitatii2",
						"afebr2", "transpir2", "diureza2", "odinofag2"]

confirmari = ["nuconf", "daconf", "nustieconf"]

calatorii = ["germania", "da", "nu", "scotia", "franta", "portugalia"]

transport = ["nut", "trent", "masinat", "aviont", "dat"]

print("Parsing excel...")

df = pd.read_excel(sys.argv[1])
# print(df[0:50])
# df.to_excel("output.xlsx")


df.replace(to_replace=";", value ="," )

for x in simptome_declarate:
	df[x] = 0

for x in simptome_raportate:
	df[x] = 0

for x in confirmari:
	df[x] = 0

for x in calatorii:
	df[x] = 0

for x in transport:
	df[x] = 0

df["rezultat testare"] = df["rezultat testare"].str.lower()

df["rezultat testare"].replace('', np.nan, inplace=True)
df["rezultat testare"].replace("neconcludent", np.nan, inplace=True)
df.dropna(subset=["rezultat testare"], inplace=True)

df["sex"] = df["sex"].str.lower()
df["instituția sursă"] = df["instituția sursă"].str.lower()

df["simptome declarate"] = df["simptome declarate"].str.lower()
df["simptome declarate"].replace(np.nan, "-", inplace=True)

df["simptome raportate la internare"] = df["simptome raportate la internare"].str.lower()
df["simptome raportate la internare"].replace(np.nan, "-", inplace=True)

df["istoric de călătorie"] = df["istoric de călătorie"].str.lower()
df["istoric de călătorie"].replace(np.nan, "-", inplace=True)

df["mijloace de transport folosite"] = df["mijloace de transport folosite"].str.lower()
df["mijloace de transport folosite"].replace(np.nan, "-", inplace=True)

df["confirmare contact cu o persoană infectată"] = df["confirmare contact cu o persoană infectată"].str.lower()
df["confirmare contact cu o persoană infectată"].replace(np.nan, "-", inplace=True)

for ind in df.index: 
	if df["sex"][ind] == "masculin":   
		df["sex"][ind] = 0
	else:
		df["sex"][ind] = 1


	if df["instituția sursă"][ind] == "x":
		df["instituția sursă"][ind] = 0
	elif df["instituția sursă"][ind] == "y":
		df["instituția sursă"][ind] = 1
	elif df["instituția sursă"][ind] == "z":
		df["instituția sursă"][ind] = 2
	else: 
		df["instituția sursă"][ind] = 3


	if df["rezultat testare"][ind] == "pozitiv":   #scos cei cu timestamp, neconcludent sau nimic
		df["rezultat testare"][ind] = 1
	else:
		df["rezultat testare"][ind] = 0

	for j in simptome_declarate:
		x = j[0:(len(j)-1)]
		if x in df["simptome declarate"][ind]:
			df[j][ind] = 1

	for j in simptome_raportate:
		x = j[0:(len(j)-1)]
		if x in df["simptome raportate la internare"][ind]:
			df[j][ind] = 1
		if "temperatura" in df["simptome raportate la internare"][ind]:
			df["febr2"][ind] = 1
		if "greturi" in df["simptome raportate la internare"][ind]:
			df["greata2"][ind] = 1
		if "nu este cazul" in df["simptome raportate la internare"][ind]:
			df["asim2"][ind] = 1
		if "nu" in df["simptome raportate la internare"][ind]:
			df["asim2"][ind] = 1

	for j in calatorii:
		if j in df["istoric de călătorie"][ind]:
			df[j][ind] = 1		
		if "neaga" in df["istoric de călătorie"][ind]:
			df["nu"][ind] = 1

	for j in transport:
		x = j[0:(len(j)-1)]
		if x in df["mijloace de transport folosite"][ind]:
			df[j][ind] = 1

	if "nu" in df["confirmare contact cu o persoană infectată"][ind] or "neaga" in df["confirmare contact cu o persoană infectată"][ind] or "neagă" in df["confirmare contact cu o persoană infectată"][ind] or "nu este cazul" in df["confirmare contact cu o persoană infectată"][ind] or "fara" in df["confirmare contact cu o persoană infectată"][ind]	or "Nu a avut contact cu nici un caz confirmat" in df["confirmare contact cu o persoană infectată"][ind]:
		df["nuconf"][ind] = 1

	if "nu se stie" in df["confirmare contact cu o persoană infectată"][ind] or "nu știe" in df["confirmare contact cu o persoană infectată"][ind] or "nu cunoaste" in df["confirmare contact cu o persoană infectată"][ind] or "nu avem informatii" in df["confirmare contact cu o persoană infectată"][ind]:
		df["nustieconf"][ind] = 1

	if "da" in df["confirmare contact cu o persoană infectată"][ind] or	"focar familial" in df["confirmare contact cu o persoană infectată"][ind]:
		df["daconf"][ind] = 1

timestamp("dată internare")
timestamp("data rezultat testare")
timestamp("dată debut simptome declarate")
s = round(medie("vârstă"))
df["vârstă"] = df["vârstă"].replace(np.nan, s)

df["sex"] = df["sex"].astype("int64")
df["rezultat testare"] = df["rezultat testare"].astype("int64")
df["instituția sursă"] = df["instituția sursă"].astype("int64")
df["vârstă"] = df["vârstă"].astype("int64")
df["dată debut simptome declarate"] = df["dată debut simptome declarate"].astype("int64")
df["dată internare"] = df["dată internare"].astype("int64")
df["data rezultat testare"] = df["data rezultat testare"].astype("int64")

print(df["rezultat testare"].value_counts())

####
dfNegativ = df[df["rezultat testare"] == 0]
dfPozitiv = df[df["rezultat testare"] == 1]

dfCursi = pd.concat([dfNegativ[0:500], dfPozitiv[0:500]])
dfCursi = dfCursi.sample(frac=1).reset_index(drop=True)
# print(dfCursi["rezultat testare"])
####

# print(df.dtypes)
# df = df.sort_values("rezultat testare")
# print(df[0:50])
# df.to_csv(sys.argv[2], index = False)
# df.to_excel("output.xlsx", index = False)

# feature_df = df[["asim1", "febr1", "tuse1", "dispn1", "tulbura1", "fatig1",
#             "convulsi1", "ameteli1", "abdo1", "muscu1", "cefalee1", "edem1", "paloare1",
#             "tumefie1", "echimoza1", "greturi1", "varsatu1", "diar1", "hta1",
#             "trombembolie1", "fris1", "odinofag1","subf1", "mialg1", "muscula1", "asteni1",
#              "epistaxis1", "inapetent1", "epigastri1", "erupt1", "transpir1", "asim2", "febr2", "tuse2", "dispn2", "abd2", "fris2", "toracic2", "asten2",
#             "fatig2", "subf2", "diare2", "greata2", "cefalee2", "edeme2", "varsatu2",
#             "mialg2", "inapetent2", "grav2", "disfagie2", "faringe2", "epigastr2","palpitatii2",
#             "afebr2", "transpir2", "diureza2", "odinofag2", "nuconf", "daconf", "nustieconf",
#             "germania", "da", "nu", "scotia", "franta", "portugalia", "nut", "trent", "masinat", "aviont", "dat",
#             "sex", "instituția sursă", "vârstă", "dată debut simptome declarate", "dată internare",
#             "data rezultat testare" ]]

feature_df = dfCursi[[
			"asim1", "febr1", "tuse1", "dispn1", "tulbura1", "fatig1",
            "convulsi1", "ameteli1", "abdo1", "muscu1", "cefalee1", "edem1", "paloare1",
            "tumefie1", "echimoza1", "greturi1", "varsatu1", "diar1", "hta1",
            "trombembolie1", "fris1", "odinofag1","subf1", "mialg1", "muscula1", "asteni1",
             "epistaxis1", "inapetent1", "epigastri1", "erupt1", "transpir1", "asim2", "febr2", "tuse2", "dispn2", "abd2", "fris2", "toracic2", "asten2",
            "fatig2", "subf2", "diare2", "greata2", "cefalee2", "edeme2", "varsatu2",
            "mialg2", "inapetent2", "grav2", "disfagie2", "faringe2", "epigastr2","palpitatii2",
            "afebr2", "transpir2", "diureza2", "odinofag2",
			"nuconf", "daconf", "nustieconf",
            "germania", "da", "nu", "scotia", "franta", "portugalia", "nut", "trent", "masinat", "aviont", "dat",
            "sex", "instituția sursă", "vârstă"
			]]

# df.to_excel("output_total.xlsx", index = False)
# feature_df.to_excel("output_test.xlsx", index = False)

print("Creating and training classifier...")

X = np.asarray(feature_df) # Pe astea vreau sa le folosesc sa fac predictia clasei (cancerigen sau nu) - variabile INDEPENDENTE
Y = np.asarray(dfCursi["rezultat testare"]) # Aci am rezultatul real cu care voi putea sa compar - variabile DEPENDENTE

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=4)
# Impartirea este facuta frumos
# print(np.count_nonzero(Y_train == 0), np.count_nonzero(Y_train == 1))
# print(np.count_nonzero(Y_test == 0), np.count_nonzero(Y_test == 1))

# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)

# classifier = KNeighborsClassifier()
# classifier = LogisticRegression()
classifier = svm.SVC(kernel="linear", gamma="auto", C=1)
# classifier = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100, random_state=4)
# classifier = RandomForestClassifier(class_weight="balanced", criterion= "gini", n_estimators=10, max_depth=2, random_state=0)
classifier.fit(X_train, Y_train)

print("Predicting...")
Y_predicted = classifier.predict(X_test)
print(classification_report(Y_test, Y_predicted))
print(confusion_matrix(Y_test, Y_predicted))

############################################################################################################
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# clf = GridSearchCV(svm.SVC(), tuned_parameters)
# clf.fit(X_train, Y_train)
# GridSearchCV(estimator=svm.SVC(), param_grid=tuned_parameters)
# print(clf.best_params_)