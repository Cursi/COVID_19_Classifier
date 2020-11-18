# COVID_19_Classifier
This is a simple sklearn project for an university homework whose purpose was to classify with a high accuracy based on a small dataset whether a person could have the COVID19 disease or not. 

General aspects: we chose the sklearn module from python because of its various functionalities for the ML implementation 
alongside with the pandas module for data manipulation due to its flexibility and ease of use. 
The dataset presented numerous human-error issues regarding the recording of information (typos, different formats) and many 
empty slots. The majority of entries depicted negative covid tests (90%), thus we used upscaling for achieving a good
accuracy for the positive patients as well. The program can be compiled as follow:

	> py classifier.py 0 excel_file -> for normal compilation
	> py classifier.py 1 excel_file -> to use the web app

Implementation:
	1. Data manipulation: the excel file is converted into a panda object.

	2. Cleaning the dataset:
		2.1 The rows with no results in column "rezultat testare" or with invalid information were dropped.
		2.2 In order to keep the relevant features, we synthesized the columns "simptome declarate", 
		"diagnostic și semne de internare" and "simptome raportate la internare" by selecting the most commonly
		found and prevalent simptoms by extracting the root of the words so they can match as many entries as 
		possible, notwithstanding the typos, diacritics, plural forms or the articles at the end of the words.
		These were kept in lists assigned for each column.

	3. Encoding:
		3.1 Each string is standardized by converting everything to lower.
		3.2 Everything is encoded in numerical values.
		3.3 For column "institutia sursa", the three values [x,y,z, none] are encoded [1,2,3,4].
		3.4 The column "varsta" is split into 4 groups encoded accordingly.
		3.5 The column "sex" is encoded with 1 and 2.
		3.6 For each of the columns "data rezultat testare", "dată internare" and "dată debut simptome declarate",
		we created 3 more columns and extracted the day, the month and the day within a week time, as integers.
		3.7 For column "rezultat testare", the results were encoded with 1 or 2.
		3.8 For each of the columns "simptome declarate", "diagnostic și semne de internare" and
		"simptome raportate la internare", new columns were created and if the initials columns contained 
		any of them, 0 or 1 were written in the corresponding cells of the newly created columns.
		3.9 For column "confirmare contact cu o persoană infectată" were created 3 more columns:
		one for missing value in the initial cell, one for a negative answer and one for anything that was a string or
		contained something different than a negative answer. Analog for columns "istoric de călătorie" and 
		"mijloace de transport folosite".

	4. Choosing the algorithm: After translating everything in numerical forms, in order to achieve a good accuracy
	for the positive patients (which previously were badly predicted by the program), we upscaled their category
	by concateneting random	rows from their group to the dataframe, resulting in having the same amount of negative
	and positive patiens. We used Gridsearch to evaluate the best parameters for our classifier. We chose RandomForestClassifier
	because ........ . It needed a small range of values for encodation (this is how it works well) and many features (columns).
	AdaBoostClassifier was used for improving the numeric methods used for prediction.

	5. Training: For the training part, we used 80% from data, the other 20% remaining for the testing.

	6. Results: 

					General aspects: we chose the sklearn module from python because of its various functionalities for the ML implementation 
alongside with the pandas module for data manipulation due to its flexibility and ease of use. 
The dataset presented numerous human-error issues regarding the recording of information (typos, different formats) and many 
empty slots. The majority of entries depicted negative covid tests (90%), thus we used upscaling for achieving a good
accuracy for the positive patients as well. The program can be compiled as follow:

	> py classifier.py 0 excel_file -> for normal compilation
	> py classifier.py 1 excel_file -> to use the web app

Implementation:
	1. Data manipulation: the excel file is converted into a panda object.

	2. Cleaning the dataset:
		2.1 The rows with no results in column "rezultat testare" or with invalid information were dropped.
		2.2 In order to keep the relevant features, we synthesized the columns "simptome declarate", 
		"diagnostic și semne de internare" and "simptome raportate la internare" by selecting the most commonly
		found and prevalent simptoms by extracting the root of the words so they can match as many entries as 
		possible, notwithstanding the typos, diacritics, plural forms or the articles at the end of the words.
		These were kept in lists assigned for each column.

	3. Encoding:
		3.1 Each string is standardized by converting everything to lower.
		3.2 Everything is encoded in numerical values.
		3.3 For column "institutia sursa", the three values [x,y,z, none] are encoded [1,2,3,4].
		3.4 The column "varsta" is split into 4 groups encoded accordingly.
		3.5 The column "sex" is encoded with 1 and 2.
		3.6 For each of the columns "data rezultat testare", "dată internare" and "dată debut simptome declarate",
		we created 3 more columns and extracted the day, the month and the day within a week time, as integers.
		3.7 For column "rezultat testare", the results were encoded with 1 or 2.
		3.8 For each of the columns "simptome declarate", "diagnostic și semne de internare" and
		"simptome raportate la internare", new columns were created and if the initials columns contained 
		any of them, 0 or 1 were written in the corresponding cells of the newly created columns.
		3.9 For column "confirmare contact cu o persoană infectată" were created 3 more columns:
		one for missing value in the initial cell, one for a negative answer and one for anything that was a string or
		contained something different than a negative answer. Analog for columns "istoric de călătorie" and 
		"mijloace de transport folosite".

	4. Choosing the algorithm: After translating everything in numerical forms, in order to achieve a good accuracy
	for the positive patients (which previously were badly predicted by the program), we upscaled their category
	by concateneting random	rows from their group to the dataframe, resulting in having the same amount of negative
	and positive patiens. We used Gridsearch to evaluate the best parameters for our classifier. We chose RandomForestClassifier
	because ........ . It needed a small range of values for encodation (this is how it works well) and many features (columns).
	AdaBoostClassifier was used for improving the numeric methods used for prediction.

	5. Training: For the training part, we used 80% from data, the other 20% remaining for the testing.

	6. Results: 

					*insert pic*
		6.1 We considered this model to be relevant due to its high accuracy and AUCROC scores.
		6.2 Accuracy: the number of well predicted cases. 
		6.3 Precision: the number of well predicted positive (or negative) cases out of the total number of
		positive (or negative) predicted cases.
		6.4 Recall: the number of well predicted positive (or negative) cases out of the total number of
		positive (or negative) cases.



		6.1 We considered this model to be relevant due to its high accuracy and AUCROC scores.
		6.2 Accuracy: the number of well predicted cases. 
		6.3 Precision: the number of well predicted positive (or negative) cases out of the total number of
		positive (or negative) predicted cases.
		6.4 Recall: the number of well predicted positive (or negative) cases out of the total number of
		positive (or negative) cases.


