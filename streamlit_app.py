
import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


    # Function to load the model
def load_model():
    # Use a raw string (prefix with 'r') or double backslashes to avoid escape character issues
    model_path = r'/content/drive/MyDrive/Grammys/Grammys2/logit_model.pkl'

    # Load the model
    model_A = pickle.load(open(model_path, 'rb'))
    return model_A

# Function to get predictions based on user input
def get_prediction(user_input, model_A):
    # Make sure user_input is a DataFrame with the same columns as your training data
    
    # Check if the model is fitted
    if hasattr(model_A, 'predict'):
        # Make predictions
        prediction = model_A.predict(user_input)
        return prediction
    else:
        raise ValueError("The model is not fitted. Please fit the model before making predictions.")

# Streamlit app

def main():
    st.title('grammy_prediction_app')

    # Collect user input for dependent variables
    Age = st.number_input("Insert artist's age", None)
    Usersscore = st.number_input("Insert the users score", None, placeholder="Type a float between 0 and 10")
    Criticscore = st.number_input("Insert the Critic score", None, placeholder="Type an int between 0 and 100")
    st.write("Select artist's ethnicity")
    Ethnicity_1 = st.checkbox('black')
    Ethnicity_2 = st.checkbox('white')
    Ethnicity_3 = st.checkbox('polynesian')
    Ethnicity_4 = st.checkbox('mixed')
    Ethnicity_5 = st.checkbox('asian')
    Ethnicity_6 = st.checkbox('latin')
    st.write("Select artist's gender")
    Gender_Female = st.checkbox('Female')
    Gender_Male = st.checkbox('Male')
    Gender_NB = st.checkbox('NB')

    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'Age of win': [Age],
        'GENDERS': ['Female' if Gender_Female else None,
                    'Male' if Gender_Male else None,
                    'NB' if Gender_NB else None],
        'Ethnicity': ['black' if Ethnicity_1 else None,
                      'white' if Ethnicity_2 else None,
                      'polynesian' if Ethnicity_3 else None,
                      'mixed' if Ethnicity_4 else None,
                      'asian' if Ethnicity_5 else None,
                      'latin' if Ethnicity_6 else None],
        'Users score': [Usersscore],
        'Critic score': [Criticscore],
    })

    # Display user input
    st.subheader('User Input:')
    st.write(user_input)

    # Load the model
    model_A = load_model()

    # Train the model (if not done before)
    # model_A.fit(X_train, y_train)  # Replace X_train and y_train with your training data

    # Get prediction
    prediction = get_prediction(user_input, model_A)

    A0_html = """
      <div style="background-color:#F4D03F;padding:10px;">
       <h2 style="color:white;text-align:center;"> Null</h2>
       </div>
    """
    A0_A1_html = """
      <div style="background-color:#F4D03F;padding:10px;">
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

