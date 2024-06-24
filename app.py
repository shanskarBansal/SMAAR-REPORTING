                                    # ------------------------------------© SHANSKAR BANSAL ©------------------------------------# 
                                    # -----------------------------        Smaar - Reporting       -------------------------------#

import pandas as pd
import re
import nltk
from concurrent.futures import ThreadPoolExecutor
from mtranslate import translate
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

def get_gspread_client():
    scope = ['https://www.googleapis.com/auth/spreadsheets']
    creds = ServiceAccountCredentials.from_json_keyfile_name('cred.json', scope)
    return gspread.authorize(creds)

def get_sheet_data(sheet_id):
    client = get_gspread_client()
    sheet = client.open_by_key(sheet_id)
    return sheet

st.title("Google Sheet Data Processor")
sheet_id = st.text_input("Enter the ID of your Google Sheet:", "")

if sheet_id:
    client = get_gspread_client()
    sheet = get_sheet_data(sheet_id)

    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    custom_tokens_to_remove = set()

    sheet_names_to_process = ['OBC X', 'KM X', 'MM X', 'KM_Insta', 'OBC_Insta', 'MM_Insta', 'MM_FB', 'OBC_FB', 'KM_FB']
    sentences_column = 'Post Caption'
    ground_truth_column = 'Post Owner'
    model = SentenceTransformer('all-mpnet-base-v2')
    selected_sheet_name = st.selectbox("Select a sheet to process:", sheet_names_to_process)
    def clean_translate_tokenize(text):
        if pd.notna(text):
            parts = text.split('#', 1)
            text_before_hash = parts[0] if len(parts) > 1 and parts[0].strip() != '' else text
        else:
            text_before_hash = ''
        
        emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]", flags=re.UNICODE)
        no_emojis = emoji_pattern.sub('', text_before_hash)
        cleaned_text = re.sub(r'[^a-zA-Z\u0900-\u097F]', ' ', no_emojis).lower()
        translated_text = translate(cleaned_text.strip(), 'en').lower()
        tokens = word_tokenize(translated_text)
        tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens if token.strip()]
        tokens = [token for token in tokens if token not in stop_words and token not in custom_tokens_to_remove]
        return ' '.join(tokens)

    def calculate_cosine_similarity(embedding1, embedding2):
        return util.cos_sim(embedding1, embedding2).item()

    def col_letter(col_index):
        if col_index < 1:
            raise ValueError("Index is too small")
        result = ""
        while col_index > 0:
            col_index, remainder = divmod(col_index - 1, 26)
            result = chr(65 + remainder) + result
        return result

    def determine_threshold(sentence):
        if len(sentence.split()) > 15:
            return 0.8
        return 0.8

    varahe_worksheet = sheet.worksheet('Captions')
    varahe_data = varahe_worksheet.get_all_records()
    varahe_df = pd.DataFrame(varahe_data)

    with ThreadPoolExecutor() as executor:
        varahe_sentences = list(executor.map(clean_translate_tokenize, varahe_df[sentences_column].fillna('')))
        varahe_embeddings = model.encode(varahe_sentences, convert_to_tensor=True)

    cleaned_sheets = {}

    columns = ['Post Caption', 'Post Owner']

    process_button = st.button('Process Selected Sheet')
    if selected_sheet_name and process_button:
        try:
            worksheet = sheet.worksheet(selected_sheet_name)
            data = worksheet.get_all_values()
            df = pd.DataFrame(data[1:], columns=data[0])
            
            if df.empty:
                st.write(f"DataFrame is empty after loading data for {selected_sheet_name}")
            else:
                st.write(f"Data loaded for {selected_sheet_name}: {df.head()}")
    
            df[sentences_column] = df[sentences_column].astype(str).fillna('')
    
            with ThreadPoolExecutor() as executor:
                row_sentences = list(executor.map(clean_translate_tokenize, df[sentences_column]))
                row_embeddings = model.encode(row_sentences, convert_to_tensor=True)
            
            results = []
            max_similarity_scores = []
            max_similarity_sentences = []
            
            for row_sentence, row_embedding in zip(row_sentences, row_embeddings):
                if row_sentence.strip():  # Only process non-empty sentences
                    best_match = ""
                    best_similarity = 0
                    best_sentence = ""
                    
                    for varahe_sentence, varahe_embedding in zip(varahe_sentences, varahe_embeddings):
                        similarity = calculate_cosine_similarity(row_embedding, varahe_embedding)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_sentence = varahe_sentence
                            best_match = "Varahe" if similarity >= determine_threshold(row_sentence) else ""
                    
                    results.append(best_match)
                    max_similarity_scores.append(best_similarity)
                    max_similarity_sentences.append(best_sentence)
                else:
                    results.append("")  # Append empty tag for empty sentences
                    max_similarity_scores.append(0)
                    max_similarity_sentences.append("")
            
            df['Model_Tags'] = results
            cleaned_sheets[selected_sheet_name] = df
    
            update_values = [df.columns.values.tolist()] + df.values.tolist()
            worksheet.update(update_values, 'A1')
            st.write('Sheet processed and updated successfully.')
    
        except Exception as e:
            st.error(f"Error processing sheet {selected_sheet_name}: {e}")
