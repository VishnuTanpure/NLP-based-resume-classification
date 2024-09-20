# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:54:28 2024

@author: Dell
"""

import os  # Provides functions to interact with the operating system
#import subprocess  # Import the subprocess module to run external system commands from Python
#from io import BytesIO  # Import BytesIO to create an in-memory binary stream for handling byte data
import pandas as pd  # Library for data manipulation and analysis
import numpy as np  # Library for numerical computing with support for large, multi-dimensional arrays
import pickle  # Used for serializing and deserializing Python objects
from pypdf import PdfReader  # For reading and extracting text from PDF files
import docx2txt  # For extracting text from DOCX files
import streamlit as st  # Streamlit for building web apps

import re  # For performing regular expression operations like pattern matching and text manipulation
import string  # Provides string constants and functions (e.g., punctuation handling)

import nltk  # Natural Language Toolkit, used for NLP tasks like tokenization, stemming, etc.
from nltk.corpus import stopwords  # For removing common stopwords (e.g., 'the', 'is', etc.)
from nltk import word_tokenize  # For tokenizing text data into individual words
from nltk.stem import PorterStemmer, WordNetLemmatizer  # For stemming and lemmatization
import contractions  # For expanding contractions like "can't" to "cannot"

# Download necessary NLTK resources (only need to be run once)
nltk.download('punkt')  # Tokenizer
nltk.download('stopwords')  # Stopwords list
nltk.download('wordnet')  # WordNet corpus for lemmatization

# Initialize NLP tools
stop_words = set(stopwords.words('english'))  # Set of English stopwords
stemmer = PorterStemmer()  # Initialize PorterStemmer for stemming
lemmatizer = WordNetLemmatizer()  # Initialize WordNetLemmatizer for lemmatization

# Load pre-trained models
word_vector = pickle.load(open("tfidf.pkl", "rb"))  # Load TF-IDF vectorizer from pickle
model = pickle.load(open("model.pkl", "rb"))  # Load the machine learning classification model (SVC in this case)

# Function to clean and preprocess resume text
def cleanResume(text):
    """Cleans and preprocesses resume text for machine learning."""
    if isinstance(text, str):
        text = contractions.fix(text)  # Expand contractions (e.g., "can't" -> "cannot")
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        text = " ".join(text.split())  # Remove extra whitespaces
        tokens = word_tokenize(text)  # Tokenize text into words
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Apply lemmatization
        tokens = [word for word in tokens if len(word) > 1]  # Remove single-character words
        return ' '.join(tokens)  # Return cleaned text
    return text

# Category mapping for predictions
category_mapping = {0: 'Peoplesoft', 1: 'Reactjs', 2: 'SQL', 3: 'Workday'}  # Maps prediction IDs to job categories

# Function to categorize resumes and save them in categorized folders
def categorize_resumes(uploaded_files, output_directory, threshold):
    """Categorizes resumes and saves them in corresponding category folders."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Create the output directory if it doesn't exist
    
    results = []  # List to store results
    
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        # Extract text from PDFs, DOCX files
        if uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)  # Initialize PdfReader for the uploaded PDF
            page = reader.pages[0]  # Read the first page
            text = page.extract_text()  # Extract text from the PDF
            
        elif uploaded_file.name.endswith('.docx'):
            text = docx2txt.process(uploaded_file)  # Extract text from DOCX file
        
        if uploaded_file.name.endswith(('.pdf', '.docx')):
            cleaned_resume = cleanResume(text)  # Clean and preprocess the extracted text

            # Transform text using the pre-trained TF-IDF vectorizer
            input_features = word_vector.transform([cleaned_resume])  # Transform resume text into feature vector
            input_features_dense = input_features.toarray()  # Convert sparse matrix to dense array
            
            # Predict class probabilities using the model
            probabilities = model.predict_proba(input_features_dense)[0]  # Get predicted probabilities
            max_prob = np.round(max(probabilities), 2)  # Get highest probability and round to two decimals
            prediction_id = model.predict(input_features_dense)[0]  # Get predicted class
            
            # Classify based on threshold
            if max_prob < threshold:
                category_name = "Other"  # Assign "Other" if probability is below threshold
                st.warning(f"Failed to classify resume no. {index}.")  # Show warning in Streamlit
                results.append({'File_Name': uploaded_file.name, 'Category': category_name, 'Probability': '< ' + str(threshold)})  # Log result
            else:
                category_name = category_mapping.get(prediction_id, "Unknown")  # Get category based on prediction ID
                st.success(f"Resume no. {index} classified as {category_name} with a probability of {max_prob}.")  # Show success message in Streamlit
                results.append({'File_Name': uploaded_file.name, 'Category': category_name, 'Probability': max_prob})  # Log result
            
            # Create category folder and save the resume
            category_folder = os.path.join(output_directory, category_name)  # Path for category folder
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)  # Create category folder if it doesn't exist
            
            target_path = os.path.join(category_folder, uploaded_file.name)  # Full path for saving the resume
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())  # Save the resume in the respective folder
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)  # Create a DataFrame for the results
    results_df.index = pd.RangeIndex(start=1, stop=len(results_df) + 1)  # Set index starting from 1
    results_df.index.name = 'Sr. No.'  # Set index name
    
    return results_df  # Return the DataFrame

# Streamlit Web App UI
def main():
    st.set_page_config(layout="wide")  # Set layout to wide
    
    # Custom CSS for background and sidebar styles
    page_bg_color = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-color: #F0F8FF; /* Sky blue color */
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0); /* Transparent header */
    }
    </style>
    """
    st.markdown(page_bg_color, unsafe_allow_html=True)  # Apply background color
    
    # Sidebar content
    with st.sidebar:
        custom_css = """
        <style>
        [data-testid="stSidebar"] {
            background-color: #365563;
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)  # Apply sidebar custom CSS
        st.title("ExcelR Project")  # Sidebar title
        st.sidebar.header("Group 5")  # Sidebar group header
        #st.text("1. Humera Sikander Shaikh\n2. Sukrta G A\n3. Affan Abutalha Chaus\n4. Priyanka Kuli\n5. Pushpak Babakumar Umare\n6. Meka Vamshi\n7. Vishnu Tanpure")  # Group members
        st.text("1. Vishnu Tanpure\n2. Meka Vamshi\n3. Affan Abutalha Chaus\n4. Sukrta G A")
        footer_html = """
        <div style='text-align: left;'>
            <p style="margin-bottom:2cm;"> </p>
            <p style="color:#000080; font-size:14px;"> 
                <b>Designed and Developed by</b><br>
                <i>Vishnu Tanpure</i><br>
                <a href="mailto:vishnutanpure@gmail.com" style="color:#000080; text-decoration:none;">vishnutanpure@gmail.com</a>
            </p>
        </div>
        """
        st.markdown(footer_html, unsafe_allow_html=True)


    # Main app content
    st.markdown("<h1 style='text-align: center;'>Resume Classification</h1>", unsafe_allow_html=True)  # Main title
    st.markdown("<h3 style='text-align: center;'>Using Support Vector Classifier</h3>", unsafe_allow_html=True)  # Sub-title

    # Create three columns, where the middle one acts as a spacer
    col1, spacer, col2 = st.columns([2.0, 0.5, 2.0])  # Layout with two main columns and one spacer

    with col1:
        uploaded_files = st.file_uploader("Upload resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)  # File uploader
        threshold = st.slider('Set classification threshold', min_value=0.0, max_value=1.0, value=0.5)  # Slider for threshold
        output_directory = st.text_input("Specify Output Directory", "categorized_resumes")  # Input for output directory
        
        if st.button("Categorize Resumes"):  # Button for triggering categorization
            if uploaded_files and output_directory:
                results_df = categorize_resumes(uploaded_files, output_directory, threshold)  # Call categorize_resumes function
                with col2:
                    st.markdown("### Classification Results")  # Display results title
                    st.write(results_df)  # Display results table
                    # Download button for the results
                    results_csv = results_df.to_csv(index=False).encode('utf-8')  # Convert DataFrame to CSV
                    st.download_button("Download results as CSV", data=results_csv, file_name='categorized_resumes.csv', mime='text/csv')  # Download button for CSV
                    st.success("Resumes have been successfully categorized!")  # Success message
            else:
                st.error("Please upload resumes and specify an output directory.")  # Error message
                
         
if __name__ == '__main__':
    main()  # Call the main function
