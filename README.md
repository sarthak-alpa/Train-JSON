# Train-JSON

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ydata_profiling import ProfileReport

df = pd.read_csv("C:/Sarthak Prompt/train_u6lujuX_CVtuZ9i.csv")
print(df.head())

# Display the first few rows of the DataFrame
print("Original DataFrame:")
print(df.head())

# Generating profile report
profile = ProfileReport(df)
profile.to_file(output_file="output.html")

# Define the list of categorical columns to be label encoded
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for column in categorical_columns:
    if column in df.columns:
        df[column] = label_encoder.fit_transform(df[column])

# Display the DataFrame after label encoding
print("\nDataFrame after Label Encoding:")
print(df.head())

# Calculate the mean values before and after label encoding
mean_before_encoding = df[categorical_columns].mean()
print("\nMean values before Label Encoding:")
print(mean_before_encoding)

mean_after_encoding = df[categorical_columns].mean()
print("\nMean values after Label Encoding:")
print(mean_after_encoding)

# Calculate accuracy of mean values after label encoding
accuracy_of_mean = (mean_after_encoding - mean_before_encoding) / mean_before_encoding * 100
print("\nAccuracy of Mean Values (%):")
print(accuracy_of_mean)
