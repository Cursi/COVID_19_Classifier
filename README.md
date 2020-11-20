# COVID_19_Classifier

### Description
This is a SKLearn project for an university homework whose purpose was to <strong>classify</strong> with a high accuracy based on a small dataset whether a person could have the <strong>COVID19</strong> disease or not. 

### Demo
Drag and drop any test dataset respecting <strong>mps.dataset.xlsx</strong> column structure <a href="https://mps-covid-19-backend.herokuapp.com/"><strong>here</strong></a> and wait :)

### Local training & test
<strong>python classifier.py 0 mps.dataset.xlsx</strong><br><br>
Be sure to lower the n_estimators to 10 in order to achieve results in seconds or higher it to 1000 or more (if your machine can handle) for better results, but longer fitting time.

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
- For column "instituția sursă", the three values [x, y, z] were encoded [1, 2, 3] and invalid values with 0.
- For column "vârstă" all values were converted to numeric and all invalid values were converted to 0.
- The column "sex" was encoded with [1, 2] and invalid values with 0.
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

### Metrics goal
We focused on maximizing the accuracy and AUCROC score <strong>at the same time</strong>.
An around 99% for both is a very good result considering the human errors and missing data.

### Metrics interpretation
- Accuracy = the number of well predicted cases
- Precision = the number of well predicted positive (or negative) cases out of the total number of positive (or negative) predicted cases.
- Recall = the number of well predicted positive (or negative) cases out of the total number of positive (or negative) cases.
- F1 score = harmonic mean between precision & recall.
- Confusion matrix = [[True positives, False positives], [False negatives, True negative]]
- AUCROC curve = a performance measurement for the classification problems at various threshold settings. ROC is a probability curve, and AUC represents the degree or measure of separability. So...AUCROC score tells how much model is capable of distinguishing between classes.


### Possible improvements
- Encoding in a useful manner the 3 unused columns mentioned in step 1.
- Grouping and encoding better the columns containing pacient symptoms. (The current solution only improves the accuracy by 1%)

### Demo environment flow
For the demo I used Heroku. <br><br>
Due to its ephemeral file system no files can be directly stored, so the input file received when dragging effect is triggered in front end is converted to base64 and passed to a nodeJS server. <br><br>
The nodeJS server passes it further as input to the classifier.py, using argument 1, knowing that it is coming from the node server as a demo, so it will load the model directly instead of training and testing for development purposes. <br><br>
Due to Heroku 30s response limitations I decided to run the prediction in background and poll at 1 second from the frontend for the results. When they are ready, the nodeJS server returns them in a json format passed back from the python script.<br><br>
The json response is parsed and displayed nicely in the frontend.
