# app.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import text_preprocessing as tp  # Import the text_preprocessing module
import model_loading as ml  # Import the model_loading module

def load_data():
    data = pd.read_csv('clustered_data.csv')
    # Convert to a GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['Lng'], data['Lat']))
    return gdf

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
def create_map(data, filtered_data, selected_equipment):
        # Set the location for the map center
        center = filtered_data.iloc[0][['Lat', 'Lng']]
        map = folium.Map(location=[center['Lat'], center['Lng']], zoom_start=12)

        # Add markers for the equipment
        for _, row in filtered_data.iterrows():
            tooltip = f"{row['Id']} - {row['Type']}"
            color = 'red' if row['Id'] == selected_equipment else 'blue'
            folium.Marker([row['Lat'], row['Lng']], tooltip=tooltip, icon=folium.Icon(color=color)).add_to(map)

        return map
def filter_data(data, selected_equipment, clustering_type):
        selected_cluster = data[data['Id'] == selected_equipment][clustering_type].values[0]
        return data[data[clustering_type] == selected_cluster]
def display_data(data):
    # Convert the geometry column to a string
    data['geometry'] = data['geometry'].astype(str)
    st.dataframe(data)

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
    st.title('Démonstration de clustering')
    data = load_data()
    equipment = st.selectbox('Choisissez un équipement :', data['Id'].unique())
    clustering_labels = {
    'Clustering de Distances': 'Cluster',
    'Clustering avec Pénalités': 'Cluster_with_penalty',
    'Clustering de Temps': 'Cluster_time',
    'Clustering Combiné': 'Cluster_time_distance'
    }
    clustering_descriptions = {
        'Cluster': 'Regroupement basé sur la distance entre les équipements. Il considère principalement la distance de conduite entre chaque équipement.',
        'Cluster_with_penalty': 'Regroupement qui intègre des pénalités. Une pénalité est ajoutée si les équipements ne sont pas sur la même route et un bonus est accordé pour ceux sur la même route.',
        'Cluster_time': 'Regroupement basé sur le temps de parcours entre les équipements. Ce type prend en compte les feux de signalisation, les arrêts, et autres facteurs affectant le temps de trajet.',
        'Cluster_time_distance': 'Combinaison de la distance et du temps entre les équipements. Ce type de clustering évalue à la fois la distance de conduite et les facteurs temporels pour regrouper les équipements.'
    }
    selected_label = st.selectbox('Sélectionnez le Type de Clustering', list(clustering_labels.keys()))
    selected_clustering_type = clustering_labels[selected_label]

    st.write(clustering_descriptions[selected_clustering_type])

    if equipment and selected_clustering_type:
        filtered_data = filter_data(data, equipment, selected_clustering_type)
        map = create_map(data, filtered_data, equipment)
        folium_static(map)

        # Display the filtered dataset
        display_data(filtered_data)
    
    precision_data = {
        'Méthode': [
            'Distance de conduite',
            'Distance de conduite avec pénalités',
            'Temps de conduite',
            'Temps et distance de conduite'
        ],
        'Précision Globale': ['51 %', '51 %', '67 %', '79 %']
    }

    precision_df = pd.DataFrame(precision_data)
    st.table(precision_df)




if __name__ == "__main__":
    main()