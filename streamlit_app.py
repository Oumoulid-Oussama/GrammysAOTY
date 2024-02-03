#Connect to google Drive
from google.colab import drive
drive.mount('/content/drive/')

#Import useful packages
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as seabornInstance
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

data1=pd.read_csv('/content/drive/MyDrive/Grammys/Grammys2/Main-data-CSV.csv.csv')
data1.head()

data1.tail()

# Delete the last 11 rows
data = data1.iloc[:-22]

data.tail()



# Calculate the mean of the 'Userscore' column
mean_userscore = data['Userscore'].mean()

# Print or use the mean value
print(f"Mean Userscore: {mean_userscore}")

import pandas as pd

# Assuming 'data' is your DataFrame
# Fill NaN values in the 'Userscore' column with 7.9
data['Userscore'].fillna(7.9, inplace=True)

# If you want to fill NaN values for the entire DataFrame, you can use:
# data.fillna(7.9, inplace=True)


data.dtypes

#Summary of data
data.describe()

#Let's build a word cloud of Artists
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Assuming 'Artist(s)' is the column containing artist names
data['Artist(s)'] = data['Artist(s)'].astype(str)

# Generate the word cloud
Cloud_Grammy_best_album = WordCloud(background_color='white').generate(''.join(data['Artist(s)']))

# Display the word cloud
plt.imshow(Cloud_Grammy_best_album, interpolation='bilinear')
plt.axis('off')
plt.show()


# Calculate the percentage of each gender
gender_percentage = data['GENDERS'].value_counts(normalize=True) * 100

# Display the result
print(gender_percentage)

import pandas as pd
# Define a mapping for the Ethnicity values
ethnicity_mapping = {
    'black': 0,
    'white': 1,
    'mixte': 2,
    'asian': 3,
    'latin': 4,
    'polynesian': 5
}

# Update the values in the 'Ethnicity' column based on the mapping
data['Ethnicity'] = data['Ethnicity'].map(ethnicity_mapping)
data['Ethnicity'].fillna(1, inplace=True)
# Save the updated DataFrame back to the CSV file
data.to_csv('/content/drive/MyDrive/Grammys/Main CSV data.csv.csv', index=False)

# Print the updated DataFrame
print(data)

import pandas as pd

# Replace 'male' with 0, 'female' with 1, and 'NB' with 2
data['GENDERS'] = data['GENDERS'].replace({'male': 0, 'female': 1, 'NB': 2})

# Fill NaN values in the 'GENDERS' column with 1
data['GENDERS'] = data['GENDERS'].fillna(1)

# Save the modified DataFrame back to the CSV file
data.to_csv('/content/drive/MyDrive/Grammys/Main CSV data.csv.csv', index=False)

# Convert True to 1 and False to 0
data['Win'] = data['Win'].map({True: 1, False: 0, np.nan: np.nan})

# Display the updated DataFrame
print(data)

#check if missing data
data.isnull().sum()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, r2_score

# Assuming you have already preprocessed the 'GENDERS' column as mentioned in the previous responses

# Split the dataset into labels and attributes
X = data[['Age of win', 'GENDERS', 'Ethnicity', 'Critic score', 'Userscore']]
Y = data['Win']

# Let's divide the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Standardize the features (optional but can be beneficial for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform Logistic Regression
logit_model = LogisticRegression(random_state=0)
logit_model.fit(X_train_scaled, Y_train)

# Predict probabilities on the test set
probabilities = logit_model.predict_proba(X_test_scaled)[:, 1]

# Set a threshold (you can adjust this based on your requirements)
threshold = 0.5

# Convert probabilities to binary predictions based on the threshold
predictions = (probabilities > threshold).astype(int)

# Evaluate accuracy
accuracy = accuracy_score(Y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Calculate R squared
#r2 = r2_score(Y_test, probabilities)
#print(f'R squared: {r2:.4f}')

# Find the index of the most likely winner
most_likely_winner_index = probabilities.argmax()

# Extract the corresponding row from the test set
most_likely_winner_data = X_test.iloc[[most_likely_winner_index]]
print('Most Likely Winner Data:')
print(most_likely_winner_data)


import pandas as pd
from scipy.spatial import distance

# Load the CSV file and preprocess 'GENDERS' as needed
csv_file_path = '/content/drive/MyDrive/Grammys/TEstGrammysfff.csv'
csv_data = pd.read_csv(csv_file_path)

# Preprocess 'GENDERS' column if not done previously
# csv_data['GENDERS'] = csv_data['GENDERS'].replace({'male': 0, 'female': 1, 'NB': 2})

# Most Likely Winner Data
most_likely_winner_data = {'Age of win': 38, 'GENDERS': 0, 'Ethnicity': 1.0, 'Critic score': 84, 'Userscore': 5.6}

# Calculate Euclidean distances between each row in the CSV file and the Most Likely Winner Data
csv_data['Distance'] = csv_data.apply(lambda row: distance.euclidean(
    [row['Age of win'], row['GENDERS'], row['Ethnicity'], row['Critic score'], row['Userscore']],
    [most_likely_winner_data['Age of win'], most_likely_winner_data['GENDERS'], most_likely_winner_data['Ethnicity'], most_likely_winner_data['Critic score'], most_likely_winner_data['Userscore']]
), axis=1)

# Find the row with the smallest distance (most similar)
most_similar_row = csv_data.loc[csv_data['Distance'].idxmin()]

# Display the most similar row
print('Most Similar Row:')
print(most_similar_row)

import pandas as pd
from scipy.spatial import distance

# Load the CSV files
most_likely_winner_data = {'Age of win': 38, 'GENDERS': 0, 'Ethnicity': 1.0, 'Critic score': 84, 'Userscore': 5.6}
csv_file_path = '/content/drive/MyDrive/Grammys/TEstGrammys SZA.csv'
additional_csv_file_path = '/content/drive/MyDrive/Grammys/TEstGrammys SZA.csv'  # Replace with the actual path

csv_data = pd.read_csv(csv_file_path)
additional_data = pd.read_csv(additional_csv_file_path)

# Preprocess 'GENDERS' column if not done previously
# csv_data['GENDERS'] = csv_data['GENDERS'].replace({'male': 0, 'female': 1, 'NB': 2})

# Calculate Euclidean distances between each row in the CSV file and the Most Likely Winner Data
csv_data['Distance'] = csv_data.apply(lambda row: distance.euclidean(
    [row['Age of win'], row['GENDERS'], row['Ethnicity'], row['Critic score'], row['Userscore']],
    [most_likely_winner_data['Age of win'], most_likely_winner_data['GENDERS'], most_likely_winner_data['Ethnicity'], most_likely_winner_data['Critic score'], most_likely_winner_data['Userscore']]
), axis=1)

# Find the row with the smallest distance (most similar)
most_similar_row = csv_data.loc[csv_data['Distance'].idxmin()]

# Extract Album and Artist(s) from the additional CSV file using the index of the most similar row
additional_info = additional_data.loc[most_similar_row.name, ['Album', 'Artist(s)']]

# Display Album and Artist(s)
print('Most Similar Album and Artist(s):')
print(additional_info)


!pip install streamlit
import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression    
model_A=logit_model
# Streamlit app

def main():
    st.title('grammy_prediction_app')

    # Collect user input for dependent variables
    Age = st.number_input("Insert artist's age", None)
    Usersscore = st.number_input("Insert the users score", None, placeholder="Type a float between 0 and 10")
    Criticscore = st.number_input("Insert the Critic score", None, placeholder="Type a int between 0 and 100")
    st.write("Select artist's ehtnicity")
    Ethnicity_1 = st.checkbox('black')
    Ethnicity_2= st.checkbox('white')
    Ethnicity_3 = st.checkbox('polynisan')
    Ethnicity_4 = st.checkbox('mixed')
    Ethnicity_5 = st.checkbox('asian')
    Ethnicity_6 = st.checkbox('latin')
    st.write("Select artist's gender")
    GENDER = st.checkbox('Female')
    GENDER = st.checkbox('Male')
    GENDER = st.checkbox('NB')

    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'Age of win': [Age],
        'GENDERS': ['Female' if st.checkbox('Female') else None],
        'GENDERS': ['Male' if st.checkbox('Male') else None],
        'GENDERS': ['NB' if st.checkbox('NB') else None],
        'Ethnicity': ['white' if st.checkbox('white') else None],
        'Ethnicity': ['asian' if st.checkbox('asian') else None],
        'Ethnicity': ['polynisian' if st.checkbox('polynisian') else None],
        'Ethnicity': ['mixed' if st.checkbox('mixed') else None],
        'Ethnicity': ['latin' if st.checkbox('latin') else None],
        'Users score': [Usersscore],
        'Critic score': [Criticscore],
        
    })

# Display user input
    st.subheader('User Input:')
    st.write(user_input)

    # Train the model (if not done before)
    # model_A.fit(X_train, y_train)  # Replace X_train and y_train with your training data

    # Get prediction
    prediction = model_A.predict(user_input)
    

    A0_html = """
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Null</h2>
       </div>
    """
    A0_A1_html = """
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> One</h2>
       </div>
    """

    # Display prediction
    if st.button("Predict"):
        output_A = prediction
        st.success('Output value {}'.format(output_A))

        if output_A == 1:
            st.markdown(A0_html, unsafe_allow_html=True)
            st.write('The model predicts a win')
        else:
            st.markdown(A0_A1_html, unsafe_allow_html=True)
            st.write("The model isn't predicting a win")

if _name_ == '_main_':
    main()
