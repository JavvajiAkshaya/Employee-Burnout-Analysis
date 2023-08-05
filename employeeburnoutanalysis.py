

import pandas
import pip
import pydrive
import sklearn



import os
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Authenticate using credentials.json
gauth = GoogleAuth()
gauth.LoadCredentialsFile("credentials.json")  # Make sure credentials.json is in the same directory

if gauth.credentials is None:
    # Authenticate if credentials are not available
    gauth.LocalWebserverAuth()

elif gauth.access_token_expired:
    # Refresh the token if it's expired
    gauth.Refresh()
else:
    # Initialize the saved credentials
    gauth.Authorize()

gauth.SaveCredentialsFile("credentials.json")  # Save the updated credentials

# Create GoogleDrive instance
drive = GoogleDrive(gauth)

# Replace FILE_ID with the ID of the database file in your Google Drive
FILE_ID = "102500211853309682631"
destination_file = os.path.join(os.getcwd(), "employee_data.csv")

# Download the database file from Google Drive
file = drive.CreateFile({'id': FILE_ID})
file.GetContentFile(destination_file)

# Load the dataset
data = pd.read_csv(destination_file)

# Separate features (X) and target (y)
X = data.drop(columns=['burnout'])
y = data['burnout']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print("Classification Report:")
print(classification_report(y_test, y_pred))
