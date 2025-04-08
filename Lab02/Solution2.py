import pandas as pd
import matplotlib.pyplot as plt

# Downloaded from https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset
csv_path = 'student_depression_dataset.csv' 

df = pd.read_csv(csv_path)

print(df.head())

# Check for missing values
print(df.isna().sum())
# Fill missing values with the mean of the column
df.fillna(df.select_dtypes(include=['float64', 'int64']).mean(), inplace=True)
for column in df.select_dtypes(include=['object']).columns:
  df[column].fillna(df[column].mode()[0], inplace=True) # Fill categorical columns with the mode

# Check for duplicates
print(df.duplicated().sum())
# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Boxplot of academic pressure
plt.boxplot(df['Academic Pressure'])
plt.ylabel('Academic Pressure')
plt.title('Boxplot of Academic Pressure')
plt.grid(axis='y', alpha=0.5)
plt.show()

# Histogram of students by age
plt.hist(df['Age'], bins=10, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Number of Students')
plt.title('Distribution of Students by Age')
plt.grid(axis='y', alpha=0.5)
plt.show()

# Pie plot of student sleep duration
sleep_duration_counts = df['Sleep Duration'].value_counts()
plt.pie(sleep_duration_counts, labels=sleep_duration_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Student Sleep Duration')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
