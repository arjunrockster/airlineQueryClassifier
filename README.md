# airlineReviewClassifier
A classifier based on machine learning techniques to classify user reviews into the different classes. It is a multi - class classification problem. 

# Packages: Below packages were used for supervised classification.
 - pandas
 - numpy
 - nltk
 - re
 - sklearn

For detailed installing instructions please refer to the following links: 
- https://www.python.org/getit/
- https://pip.pypa.io//en/latest/

# Approach taken (Pipelines)
 - Data Preprocessing
 - Stratified Split
 - Modeling
 - Model Evaluation
 
# Dataset
The dataset has 398 rows and 2 columns (line, class). Line indicates the user query and the class indicates the type of query. This is a pre-tagged (labeled) dataset.

# Data Preprocessing
The dataset has a total of 7 classes with the below splits. 

#####################
- login        105     
- other         79
- baggage       76
- check in      61
- greetings     45
- cancel        16
- thanks        16
######################

1. A brief examination of the data reveals that the words used by users for the same kind of query varies. For instance, check-in has been addressed as check-in, check in, checkin, check me in. Thus, it becomes sensible to replace these different version with a predefined text (This will reduce the sparsity). 
2. Further, we can remove the stop words again to minimize sparsity of the resulting matrix. 
3. We can then use stemming / lemmatization (try both and see which will be better for us).

# Stratified Split
Once we are done pre-processing our data, we will split the data using stratified split. Stratified Split will give better results in this case as the data we have is not evenly split. (Login has 105 records while Thanks and Cancel classes have only 16 records) 

# Modeling
We will then model using different algorithms. We will try -
1. RandomForestClassifier
2. PassiveAggressiveClassifier
3. LinearSVC
4. Stochastic Gradient Descent (SGD)

(We will give a try for OnevsMultiple and OnevsOne to see which performs better)

# Model Evaluation
Since ours is a multi-class classification problem, accuracy will not be a good evalutor. Thus, we will have the confusion matrix, precision, recall and the F-score as our evaluation metrics. 

# Iterations
Based on the confusion matrix, we will try to see if we can introduce any other pre-processing steps and see if the model performs better. 
