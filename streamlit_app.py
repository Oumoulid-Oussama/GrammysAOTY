
import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

model_A=pickle.load(open('/content/drive/MyDrive/Grammys/Grammys2/model_Aa.pkl','rb'))

 

def get_prediction(user_input) :

    prediction=model_A.predict(user_input)

    return prediction

# Streamlit app

def main():
    st.title('grammy_prediction_app')

    # Collect user input for dependent variables
    Age = st.number_input("Insert artist's age", value=20)
    Usersscore = st.number_input("Insert the users score", value=7.9, placeholder="Type a float between 0 and 10")
    Criticscore = st.number_input("Insert the Critic score", value=77, placeholder="Type a int between 0 and 100")
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
        'GENDERS': [1 if st.checkbox('Female') else 0],
        'GENDERS': [0 if st.checkbox('Male') else 0],
        'GENDERS': [2 if st.checkbox('NB') else 0],
        'Ethnicity': [0 if st.checkbox('black') else 1],
        'Ethnicity': [1 if st.checkbox('white') else 1],
        'Ethnicity': [3 if st.checkbox('asian') else 1],
        'Ethnicity': [5 if st.checkbox('polynisian') else 1],
        'Ethnicity': [2 if st.checkbox('mixed') else 1],
        'Ethnicity': [4 if st.checkbox('latin') else 1],
        'Users score': [Usersscore],
        'Critic score': [Criticscore],
        
    })

# Display user input
    st.subheader('User Input:')
    st.write(user_input)

    # Train the model (if not done before)
    # model_A.fit(X_train, y_train)  # Replace X_train and y_train with your training data

    # Get prediction
    prediction = get_prediction(user_input)

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

if __name__ == '__main__':
    main()
