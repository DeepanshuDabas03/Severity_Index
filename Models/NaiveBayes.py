# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
d=pd.read_csv("Data/Cleandata.csv")
# Index(['Unnamed: 0.1', 'Unnamed: 0', 'Year', 'Severity', 'Start_Lat',
#        'Start_Lng', 'Distance(mi)', 'Street', 'City', 'County', 'State',
#        'Airport_Code', 'Temperature(F)', 'Wind_Chill(F)', 'Visibility(mi)',
#        'Wind_Direction', 'Weather_Condition', 'Traffic_Signal',
#        'Sunrise_Sunset', 'TimeDiff'],
#       dtype='object')

X=d[['Year',  'Start_Lat','Start_Lng', 'Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Visibility(mi)','Traffic_Signal', 'TimeDiff']]
y=d['Severity']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#bernouli
naive_bayes_classifier = BernoulliNB()
# multinomial
# naive_bayes_classifier = MultinomialNB()



# Train the model on the training data
naive_bayes_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = naive_bayes_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
    

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)
