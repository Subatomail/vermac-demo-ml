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
        
        result_data = {
            'Model': [],
            'Classe Prédite': [],
            'Certitude du choix': []
        }
        
        for model_name, model_data in models.items():
            preprocessor = model_data.get('preprocessor', None)
            input_text = preprocessor(corrected_input) if preprocessor else corrected_input
            if model_name == 'FastText':
                prediction, probability = classify_text_fasttext(input_text, model_data['model'])
            else:
                prediction, probability = classify_text(input_text, model_data['model'], model_data['vectorizer'])
            result_data['Model'].append(model_name)
            result_data['Classe Prédite'].append(f"{'Une WZ !' if prediction else 'Pas une WZ'}")
            result_data['Certitude du choix'].append(f"{probability:.2f}")
        
        result_df = pd.DataFrame(result_data)
        
        # Display the results in a table
        st.table(result_df)
        st.write('NB:')
        st.markdown(
        """
        * SVM classifie par rapport à une seul classe donc :
            * Si la certitude est proche de 1 alors cela veut dire que la phrase fait partie de la classe WZ.
            * Si la certitude est proche de 0 alors cela veut dire que la phrase ne fait pas partie de la classe WZ.
        * Pour FastText, ça donne le pourcentage de certitude que la phrase appartient à la classe prédite.
        """
        )
if __name__ == "__main__":
    main()