import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
import os
from constants import openai_key
import tensorflow as tf

# from streamlit import SessionState

st.set_page_config(page_icon='🤖')

# submit = st.empty()
# explain_tfidf = st.empty()

# ss = st.session_state.get('submit', False)


# To download the Punkt tokenizer model
nltk.download('punkt')

nltk.download('punkt_tab')

# To download the list of stopwords
nltk.download('stopwords')

# To download the WordNet lexical database
nltk.download('wordnet')

# To define a function for text preprocessing
def preprocess_text(text):

    # To convert the text to lowercase
    text = text.lower()

    # To remove HTML tags from the text
    text = re.sub('<.*?>', '', text)

    # To remove URLs from the text
    text = re.sub(r'http\S+', '', text)

    # To remove special characters and numbers from the text
    text = re.sub('[^a-zA-Z\s]', '', text)

    # To tokenize the text
    tokens = word_tokenize(text)
    
    # To remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # To lemmatize each token
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # To rejoin tokens into a single string
    processed_text = ' '.join(tokens)

    return processed_text


# Importing models, tf idf vectorizer
lr = pickle.load(open('LogisticRegression.pkl', 'rb'))
nb = pickle.load(open('NaiveBayes.pkl', 'rb'))
try:
    rnn = pickle.load(open('RNN.pkl', 'rb'))
except:
    rnn = tf.keras.models.load_model('RNN.h5')
tfidf = pickle.load(open('TF-IDF_features.pkl', 'rb'))

# y_pred = lr.predict('Hello')

# print(tfidf)

#Takes vectorized text and N as input and outputs top N words with their respective TF-IDF score
def topN_tfidf_scores(response, N):
    feature_array = np.array(tfidf.get_feature_names_out())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:N]
    tfidf_scores = response.toarray().flatten()
    top_n_scores = tfidf_scores[tfidf_sorting][:N]

    topN_tfidf_data = pd.DataFrame({'Word': top_n, 'TF-IDF Score': top_n_scores})
    return topN_tfidf_data

#Designing the prompt
def create_explanation_prompt(input_text, tfidf_data, topN_tfidf_data, model_name, prediction):
    """
    Create a prompt for the GPT model to generate an explanation.

    Parameters:
    input_text (str, mandatory): The input text provided by the application user, who wants to know if the provided text is AI generated or Human written.
    tfidf_data (DataFrame): TF-IDF scores for all words in the input text.
    topN_tfidf_data (DataFrame): TF-IDF scores for top 100 words in the input text.

    Returns:
    str: The generated prompt.
    """

    # processed_text = preprocess_text(txt)
    # input_text = [processed_text]
    # vectorized_text = tfidf.transform(input_text)

    # #predicting using Logistic Regression
    # prediction = lr.predict(vectorized_text)

    # Example confidence score (modify according to your model output)
    try:
        confidence = model_name.predict_proba(vectorized_text)[0]
    except:
        confidence = 'Yet to calculate'

    # Create a formatted string where each column name is followed by its value
    # tfidf_string = ', '.join([f"{col}={value}" for col, value in tfidf_data.items()])
    # topN_tfidf_string = ', '.join([f"{col}={value}" for col, value in topN_tfidf_data.items()])

    tfidf_string = ', '.join([f"{col}={value}" for col, value in tfidf_data.items()])
    topN_tfidf_string = ', '.join([f"{col}={value}" for col, value in topN_tfidf_data.items()])

    # coeff_string = ', '.join([f"{col}={value}" for col, value in coeff_data.items()])
    # shap_string = ', '.join([f"{col}={value}" for col, value in shap_data.items()])

    # tfidf_string = ', '.join([f"{row['Word']}={row['TF-IDF Score']}" for _, row in tfidf_data.iterrows()])

    prompt = f"""
            YOU ARE AN ASSISTANT TO THE USER OF TEXT CLASSIFICATION APPLICATION.

            Here are the details for a machine learning model prediction the assistant is assisting:
            - Input Text: {input_text}
            - TF-IDF scores for all words in the input text: {tfidf_string} 
            - Machine learning model name that user selected: {model_name}
            - Model's Predicted Class: {prediction}
            - Model's confidence score for the prediction: {confidence}
            - If the model prediction is '1' it's an AI Generated text, else it's a Human Written text.
            - TF-IDF scores for top 100 words in the input text: {topN_tfidf_string}



            Based on this information, explain to the user in non-technical terms:
            1. Provide summary of the provided text.
                - A short concise summary about the user selected model and model's prediction under 50 words.
            2. Identify the top reasons point wise for the model's predicted value from the individual tf-idf score contribution. Provide a brief explanation (within 50 words for each) of why these
                features significantly influence the final prediction.


            Remember:
            - TF-IDF score reflects the importance of a word to a document in the given text.
            2. What is the Logistic Regression model's predicted class and confidence score?
            In prediction, confidence score format.
            - The magnitude of a TF-IDF score value indicates the strength of a feature's influence on the prediction.
            - **Positive TF-IDF score values increase the likelihood of classification; negative values decrease it.**
            - Recommendations should be strictly based on the information provided for the input text and model information.
            """


    return prompt

def agent_response(input_text, model_name, prediction):

    """
    Generate an agent response using the GPT model.

    Parameters:
    input_text (str, mandatory): The input text provided by the application user, who wants to know if the provided text is AI generated or Human written.

    Returns:
    None
    """
    # processed_text = preprocess_text(txt)
    # input_text = [processed_text]
    # vectorized_text = tfidf.transform(input_text)

    tfidf_data = topN_tfidf_scores(vectorized_text, 500)
    topN_tfidf_data = topN_tfidf_scores(vectorized_text, 100)

    response=chain.invoke(create_explanation_prompt(input_text, tfidf_data, topN_tfidf_data, model_name, prediction))
    parts = response.split('\n\n')

    return parts

#Storing my API key in the environment variable
os.environ["OPENAI_API_KEY"] = openai_key

# Defining my LLM Instance and creating chain
chat_model = AzureChatOpenAI(temperature=0,
                 api_version="2024-05-01-preview",  # or your api version
                 azure_endpoint="https://projectdemo.openai.azure.com/",
                 azure_deployment="gpt-35-turbo")

parser = StrOutputParser()

chain = (
    chat_model
    | StrOutputParser()
)

# print(df)
st.title('Welcome to Text Classifier')
txt = st.text_area('# Input Your Text')

model = st.selectbox('Please select the model of your choice',
                     ('Logistic Regression', 'Naive Bayes', 'RNN'))

if model == 'Logistic Regression':
    selected_model = lr
elif model == 'Naive Bayes':
    selected_model = nb
else:
    selected_model = rnn

# flag = False

# if submit.button('Submit'):
#     ss.submit = True

# Initialize session state for the 'explain' button visibility
if 'show_explain_button' not in st.session_state:
    st.session_state.show_explain_button = False

# Preparing input text to feed the model
processed_text = preprocess_text(txt)
input_text = [processed_text]
vectorized_text = tfidf.transform(input_text)
if model == 'RNN':
    vectorized_text_rnn = vectorized_text.toarray()
    # print(vectorized_text.shape)
    rnn_input = vectorized_text_rnn.reshape(vectorized_text_rnn.shape[0],1, vectorized_text_rnn.shape[1])
    prediction = selected_model.predict(rnn_input)
    prediction = (prediction > 0.5).astype(int)[0]
else:
    prediction = selected_model.predict(vectorized_text)
# time.sleep(1)
# if model == 'RNN':
#     prediction = (prediction > 0.5).astype(int)[0]
output_text = agent_response(vectorized_text, selected_model, prediction)

if st.button('Submit'):
    if txt == '':
        st.error('Please provide your input text to get the predicted result :heavy_exclamation_mark::heavy_exclamation_mark:')
    else:
        with st.spinner('Your input data is getting preprocessed and vectorized..:arrow_upper_right::arrow_lower_left:'):
            # processed_text = preprocess_text(txt)
            # input_text = [processed_text]
            # vectorized_text = tfidf.transform(input_text)
            time.sleep(1)
            # st.write(f"Your text is: {vectorized_text}, selected model is : {model}")
        with st.spinner('Your model is predicting..:muscle:'):
            # if model == 'RNN':
            #     vectorized_text = vectorized_text.toarray()
            #     # print(vectorized_text.shape)
            #     rnn_input = vectorized_text.reshape(vectorized_text.shape[0],1, vectorized_text.shape[1])
            #     vectorized_text = rnn_input
            # prediction = selected_model.predict(vectorized_text)
            # output_text = agent_response(vectorized_text, model, prediction)
            # time.sleep(1)
            # if model == 'RNN':
            #     prediction = (prediction > 0.5).astype(int)[0]

            if prediction[0] == 0:
                st.success("It's a Human written text :male-astronaut::male-scientist:")
            elif prediction[0] == 1:
                st.error("It's an AI generated text :robot_face:")
            # time.sleep(1)
            # st.write(f"You have selected {model} model, it's a {prediction[0]}")
            # st.snow()
            st.toast(f"{model} model's prediction is on the screen!", icon='🥂')
            st.session_state.show_explain_button = True

# prediction = selected_model.predict(vectorized_text)
    
if st.session_state.show_explain_button:
    # if st.button('Explain'):    
    if st.button('Explain using TF-IDF :face_with_one_eyebrow_raised:'):
        # st.write("Hello World!")
        # output_text = ['1. Summary: The input text discusses the benefits and challenges of social media, including economic opportunities, privacy concerns, and mental health issues. It also mentions the potential impact of future technologies like augmented reality and decentralized platforms.', '2. Top 3 reasons for the model\'s predicted value:\n- "Social" and "medium" are the two words with the highest TF-IDF scores, indicating that they are the most important features in the text. This makes sense, as the text is specifically about social media and its impact.\n- "Privacy" is another important feature, with a high TF-IDF score. This suggests that the model is sensitive to the potential risks associated with social media use, such as data breaches and misuse of personal information.\n- "Mental health" is also a significant feature, with a relatively high TF-IDF score. This reflects the text\'s discussion of the negative impact that social media can have on mental health, including feelings of inadequacy, anxiety, and cyberbullying.']
        for line in output_text:
            # print(line)
            st.write(line)
    # st.session_state.show_explain_button = False