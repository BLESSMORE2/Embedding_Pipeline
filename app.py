import streamlit as st
from gensim.models import FastText
import re
from gensim.utils import simple_preprocess
import time
import os
import zipfile
import io
import tempfile
import numpy as np

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return simple_preprocess(text)

# Function to read and preprocess the corpus from an uploaded file
def read_corpus(file):
    for line in file:
        yield preprocess_text(line.decode('utf-8'))

# Function to zip the model files in memory
def zip_model(model):
    # Create a BytesIO object to hold the zip file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save the model to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save(os.path.join(temp_dir, "fasttext_model.model"))
            model.wv.save(os.path.join(temp_dir, "fasttext_model_vectors.kv"))
            
            # Explicitly save vectors and ngrams if needed
            np.save(os.path.join(temp_dir, "fasttext_model.model.wv.vectors_ngrams.npy"), model.wv.vectors_ngrams)
            np.save(os.path.join(temp_dir, "fasttext_model_vectors.kv.vectors_ngrams.npy"), model.wv.vectors_ngrams)
            
            # Zip all files in the temp_dir
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=temp_dir)
                    zipf.write(file_path, arcname=arcname)
    
    zip_buffer.seek(0)  # Rewind the buffer
    return zip_buffer

# Streamlit app
def main():
    st.title("FastText Word Embedding Trainer")
    
    # Upload cleaned text data
    uploaded_file = st.file_uploader("Upload Cleaned Text File", type=["txt"])
    
    if uploaded_file is not None:
        # Select embedding dimensions
        vector_size = st.number_input("Select Embedding Dimensions", min_value=10, max_value=500, value=50, step=10)
        
        # Train button
        if st.button("Train FastText Model"):
            try:
                # Read and preprocess the corpus
                sentences = list(read_corpus(uploaded_file))
                
                # Train FastText model
                start_time = time.time()
                model = FastText(
                    sentences,
                    vector_size=vector_size,
                    window=7,
                    min_count=5,
                    workers=4,
                    sg=1,
                    epochs=100,
                    bucket=2000000,
                    min_n=3,
                    max_n=6
                )
                end_time = time.time()
                
                # Calculate the elapsed time
                elapsed_time = end_time - start_time
                st.write("Time taken: {:.2f} minutes".format(elapsed_time / 60))
                
                st.write("Model trained successfully!")
                
                # Zip the model files in memory
                zip_buffer = zip_model(model)
                
                # Provide download link
                st.download_button(
                    label="Download Model",
                    data=zip_buffer,
                    file_name="fasttext_model.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Check the server logs for more details.")

if __name__ == "__main__":
    main()
