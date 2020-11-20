# COVID_19_Classifier

### Description
This is a SKLearn project for an university homework whose purpose was to <strong>classify</strong> with a high accuracy based on a small dataset whether a person could have the <strong>COVID19</strong> disease or not. 

### Demo
Drag and drop any test dataset respecting <strong>mps.dataset.xlsx</strong> column structure <a href="https://mps-covid-19-backend.herokuapp.com/"><strong>here</strong></a> and wait :)

### Local training & test
<strong>python classifier.py 0 mps.dataset.xlsx</strong><br><br>
Be sure to lower the n_estimators param from 1000 to 10 in order to achieve results in seconds.

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
- For each of the columns "simptome declarate", "diagnostic și semne de internare" and "simptome raportate la internare", new columns were created and if the initials columns contained any of them, 0 or 1 were written in the corresponding cells of the newly created columns. (Since there were thounds of unique values, they were humanly grouped into 10-15 categories for each, favoring predicting the positive case, because it's in minority)
#### 3. Force converting datatypes
In order to use the encoded data all columns should be treated as int64, so we force converted all columns that were seen as object before or after the encodings.
#### 4. Balancing the dataset
Because it is an unbalanced classification problem we have to balance somehow the dataset. The chosen approach was to upscale the positive dataset with the number of the negative ones, generating a new dataset of around 10500 rows, containing 50% negative cases and 50% positive cases.
This will help a lot with the learning rate. More cases to train, more chances to fail, more chances to learn. :)
#### 5. Selecting features
The features used were the onces described in step 2.
#### 6. Spliting the upsampled dataset
The newly created dataset was splitted into a 90% training dataset and a 10% testing dataset.
#### 7. Training the model
After tons of research and tries we decided to use RandomForestClassifier with 100 estimators and entropy criterion. Parameters were chosen with Grid Search.
On top of RFC we use an AdaBoostClassifier to help us more tweak other internal parameters.

### Results
              precision    recall  f1-score   support

           0       1.00      0.97      0.99       517
           1       0.98      1.00      0.99       529

    accuracy                           0.99      1046
    
    Confusion matrix:
    [[504  13]
    [  0 529]]
    AUCROC score: 0.9874274661508704

