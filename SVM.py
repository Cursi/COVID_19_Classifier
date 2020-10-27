#### Pentru ca aceste import-uri sa mearga rulati comenzile de pip3 din README
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm

#### Vom citi un CSV (TODO: De convertit XLSX-ul primit in CSV).
#### Pentru moment vom folosi un CSV de pe net pentru calcularea inferentei cancerului 
#### (problema identica cu ce vrem sa facem noi):
cell_df = pd.read_csv("./datasets/cell_samples.csv")

#### Puteti vedea detalii despre dataset:
# print(cell_df.head())
# print(cell_df.tail())
# print(cell_df.shape)
# print(cell_df.size)
# print(cell_df.count())

#### Puteti vedea separarea rezultatelor: 458 necancerigen, 241 cancerigen:
# print(cell_df["Class"].value_counts())

#### Ca sa facem plot-uri pe baza datelor putem folosi filtre (2 in csv inseamna benign, 4 malignant):
# benign_df = cell_df[cell_df["Class"] == 2]
# malignant_df = cell_df[cell_df["Class"] == 4]

#### Putem sa imbinam mai multe plot-uri intr-unul inainte sa afisam plot-ul.
#### TODO: De jucat cu datele sa vedem cam cum arata plot-urile in functie de diverse atribute.
# axes = benign_df.plot(kind="scatter", x="Clump", y="UnifSize", color="blue", label="Benign")
# malignant_df.plot(kind="scatter", x="Clump", y="UnifSize", color="red", label="Malignant", ax=axes)
# plt.show()

#### Gasirea atributelor ce nu sunt de tip numeric.
#### (Se observa ca folosind dataset-ul pentru cancer avem BareNuc de tip obiect):
# print(cell_df.dtypes)

#### Acuma...ce naiba facem cu el? Il folosim sau nu?
#### Daca il putem converti in ceva numeric, atunci da, altfel nu.
#### Conversia in numeric se poate face folosind pandas sau alt pachet py
#### (TODO: De cautat conversii daca e cazul la dataset-ul de COVID)
cell_df = cell_df[pd.to_numeric(cell_df["BareNuc"], errors="coerce").notnull()]
#### Pentru parametrul care determina comportamentul in caz de eroare:
#### TODO: alegerea corecta a lui: 
#### https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html
#### In cazul actual, "coerce" seteaza drept NaN orice nu poate converti si pastrez in dataset doar ce nu e NaN

#### Totusi, desi am convertit valorile, atributul e vazut tot ca obiect, asa ca trebuie sa facem force cast:
cell_df["BareNuc"] = cell_df["BareNuc"].astype("int64")

#### Iar acum observam ca avem toate campurile in format int64, which is great news.
# print(cell_df.dtypes)

#### Vizualizarea tuturor coloanelor (desi le putem vedea direct in CSV):
# print(cell_df.columns)

#### Selectarea coloanelor dorite:
feature_df = cell_df[["Clump", "UnifSize", "UnifShape", "MargAdh", "SingEpiSize", "BareNuc", "BlandChrom", "NormNucl", "Mit"]]
#### A se observa ca am sters coloana ID (pentru ca nu era relevanta pentru ML, 
#### respectiv coloana Class pentru ca acolo era deja raspunsul pe care vrem sa facem predictie)
# print(feature_df.columns)
#### TODO: De vazut pe dataset-ul de COVID ce putem converti si folosi numeric si ce nu folosim din dataset.

#### Separarea in variabile dependente si independente:
X = np.asarray(feature_df) # Pe astea vreau sa le folosesc sa fac predictia clasei (cancerigen sau nu) - variabile INDEPENDENTE
Y = np.asarray(cell_df["Class"]) # Aci am rezultatul real cu care voi putea sa compar - variabile DEPENDENTE
#### TODO: De facut separarea similara, unde singura variabila dependenta va fi rezultatul testului PCR

#### Impartirea subset-ului primit in 2 bucati: train si test.
#### O impartire cliseic folosita este: 80% train, 20% test:
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=4)

#### Putem vedea impartirea facuta dimensional astfel:
# print("Train: " + str(X_train.shape), "Test: " + str(X_test.shape))
#### La COVID rata reala de infectare este de 5%, asadar nu putem imparti random pentru ca avem
#### riscul de a avea prea multe sau prea putine infectari in oricare din bucati, asadar:
#### TODO: De impartit in mod relevant dataset-ul in cele 2 bucati, astfel incat fiecare bucata
#### sa contina 95% cazuri neinfectate, 5% cazuri infectate.

#### Modelam in maniera SVM (Support Vector Machine):
classifier = svm.SVC(kernel="poly", degree=5, gamma="auto", C=2)
#### kernel = functia de proiectie a hyperplanului generat pe baza datelor.
#### (Complicat, nu stiu nici eu sa explic foarte bine, insa e ideal sa alegem ceva polinomial de grad
####  cat mai mare pentru o predictie cat mai buna, insa va dura mai mult timp la rulare)
#### gamma = coeficient pentru kernel (pentru calculele din spate), cel mai probabil va ramane "auto"
#### C = costul unei predictii gresite (cat de mult se modifica hyperplanul cand se face o predictie gresita)
#### sau mai simplu spus...cat de mult pedepsim modelul cand greseste xD
#### TODO: De ales parametri corect pentru modelul de SVM in concordanta cu datele din COVID dataset

#### Bagam datele de train in classifier:
classifier.fit(X_train, Y_train)

#### Calculam predictiile pe baza datelor de test (cele 20%):
Y_predicted = classifier.predict(X_test)

#### Comparam cu rezultatul real din Y_test sa vedem cat de bine s-a descurcat classifier-ul:
print(classification_report(Y_test, Y_predicted))
#### TODO: De explicat in ceva documentatie ce inseamna fiecare metrica folosita.