import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.title("Sentiment Analyzer")

@st.cache_data
def get_all_data():
    root = "data/"
    data = []

    for file_name in ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]:
        with open(root + file_name, "r") as txtf:
            lines = txtf.read().split("\n")
            data.extend(lines)  
    return data



@st.cache_data
def preprocessingdata(all_data):
    processed = []
    for line in all_data:
        line = line.strip()
        if line == "":
            continue
        parts = line.rsplit(None, 1)  # ← ✅ meilleure séparation
       # st.write("Ligne brute :", line)
       # st.write("Découpée en :", parts)
        if len(parts) == 2 and parts[1] in ["0", "1"]:
            text, label = parts[0], int(parts[1])
            processed.append([text, label])
   # st.write("Nombre de lignes valides :", len(processed))
    return processed



all_data = get_all_data()

#st.write("Nombre total de lignes brutes :", len(all_data))
data_list = preprocessingdata(all_data)
df = pd.DataFrame(data_list, columns=["text", "label"])
#st.dataframe(df)

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

model= LogisticRegression()
model.fit(X_train,y_train)

# Initialiser session_state pour historique
if "history" not in st.session_state:
    st.session_state.history = []


user_input= st.text_input("Entre Your Review", "")

if st.button("predict sentement"):

    if user_input.strip():

        user_vect = vectorizer.transform([user_input])
        predict= model.predict(user_vect)[0]

        if predict==1:
             st.success("Merci pour votre avis positif ")
        else:
            st.error("Merci pour votre retour. Nous en tiendrons compte pour nous améliorer.")

        # Sauvegarder dans l’historique
        st.session_state.history.append({
            "text": user_input,
            "sentiment": predict
        })

if st.session_state.history:
    st.subheader("📊 Statistiques d'opinion (session en cours)")

    df = pd.DataFrame(st.session_state.history)

    # Nombre total
    total = len(df)
    positives = df['sentiment'].sum()
    negatives = total - positives
    percent_positive = positives / total * 100
    percent_negative = 100 - percent_positive

    # Affichage métriques
    st.metric("Total d’avis", total)
    st.metric("Avis positifs", f"{percent_positive:.1f} %")
    st.metric("Avis négatifs", f"{percent_negative:.1f} %")

    # Graphique camembert
    fig, ax = plt.subplots()
    ax.pie(
        [positives, negatives],
        labels=["Positifs", "Négatifs"],
        autopct='%1.1f%%',
        colors=['green', 'red']
    )
    st.pyplot(fig)

    # Historique affiché sous forme de tableau
    st.subheader("📝 Historique des avis")
    df_display = df.replace({1: "Positif", 0: "Négatif"})
    st.dataframe(df_display, use_container_width=True)        