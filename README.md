# COVID_19_Classifier

### Description
This is a SKLearn project for an university homework whose purpose was to <strong>classify</strong> with a high accuracy based on a small dataset whether a person could have the <strong>COVID19</strong> disease or not. 

### Demo
Drag and drop any test dataset respecting <strong>mps.dataset.xlsx</strong> column structure <a href="https://mps-covid-19-backend.herokuapp.com/"><strong>here</strong></a> and wait :)

### Local training & test
<strong>python classifier.py 0 mps.dataset.xlsx</strong><br><br>
Be sure to lower the n_estimators param from 1000 to 10 in order to achieve results in seconds, not hours. The model used for demo was trained for 2 hours on my personal server with 1000 estimators.

### Dependencies
python 3.6 or higher
pip3

### Pip modules
Can be found in <strong>requirements.txt</strong>

### General aspects
- The dataset presented numerous human-error issues regarding the recording of information (typos, different formats) and many empty slots.
- The majority of entries depicted negative covid tests (90%) so this is an unbalanced classification problem.
- The column used for clasification is "rezultat testare"

### Implementation
#### 1. Dataset cleanup
- Rows with no results in column "rezultat testare" or with invalid information were dropped.
- Columns "confirmare contact cu o persoană infectată", "istoric de călătorie" and "mijloace de transport folosite" were discarded due to the abundent missing values and human errors.
#### 2. Columns manipulation & encoding
- All strings were lowered.
- Everything was encoded in numerical values.
- For column "instituția sursă", the three values [missing, x, y, z] were encoded [0, 1, 2, 3].
- For column "vârstă" all values were converted to numeric and all invalid values were converted to 0.
- The column "sex" was encoded with [1, 2].
- For each of the columns "data rezultat testare", "dată internare" and "dată debut simptome declarate",
	we created 3 more columns and extracted the day, the month and the day within a week time, as integers.
- For column "rezultat testare", the results were encoded with 1 or 2. (Just for the beauty of encoding, not needed since it's the object we classify)
- For each of the columns "simptome declarate", "diagnostic și semne de internare" and
	"simptome raportate la internare", new columns were created and if the initials columns contained 
	any of them, 0 or 1 were written in the corresponding cells of the newly created columns. 
  (Since there were thounds of unique values, they were humanly grouped into 10-15 categories for each, favoring predicting the positive case, because it's in minority)
    were created 3 more columns:
	one for missing value in the initial cell, one for a negative answer and one for anything that was a string or
	contained something different than a negative answer. Analog for columns "istoric de călătorie" and 
	"mijloace de transport folosite".
