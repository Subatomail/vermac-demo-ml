# app.py

import streamlit as st
import pandas as pd
import text_preprocessing as tp  # Import the text_preprocessing module
import model_loading as ml  # Import the model_loading module


def classify_text(text, model, vectorizer):
    # Vectorize the user input
    X = vectorizer.transform([text])
    # Make and return prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]  # Assuming the positive class is labeled as 1
    return prediction, probability

def classify_text_fasttext(text, model):
    labels, probabilities = model.predict(text)
    prediction = True if '__label__True' in labels else False
    probability = probabilities[0]
    return prediction, probability

def main():
    tp.ensure_nltk_data()
    models  = ml.load_models()
    
    st.title('Démonstration de classificateur NLP')
    user_input = st.text_input("Saisissez un message d'un PCMS:")
       
    
    if user_input:
        corrected_input = tp.correct_Levenshtein(tp.correct_spelling(user_input), tp.new_words)
        print(corrected_input)
        
        result_data = {
            'Model': [],
            'Prediction': [],
            'Probability': []
        }
        
        for model_name, model_data in models.items():
            preprocessor = model_data.get('preprocessor', None)
            input_text = preprocessor(corrected_input) if preprocessor else corrected_input
            print('inside')
            print(input_text)
            if model_name == 'FastText':
                prediction, probability = classify_text_fasttext(input_text, model_data['model'])
            else:
                prediction, probability = classify_text(input_text, model_data['model'], model_data['vectorizer'])
            result_data['Model'].append(model_name)
            result_data['Prediction'].append(f"{'Une WZ !' if prediction else 'Pas une WZ'}")
            result_data['Probability'].append(f"{probability:.2f}")
        
        result_df = pd.DataFrame(result_data)
        
        # Display the results in a table
        st.table(result_df)

if __name__ == "__main__":
    main()