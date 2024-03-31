import streamlit as st
import pickle
import pandas as pd
from sklearn import *

base_data = pd.read_csv("DSP_1.csv")

filename = "model.sv"
model = pickle.load(open(filename, 'rb'))

pclass_d = {0: "Pierwsza", 1: "Druga", 2: "Trzecia"}
embarked_d = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}
sex_d = {0: "male", 1: "female"}


def main():
    st.set_page_config(page_title="Titanic survival prediction")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://facts.net/wp-content/uploads/2023/06/37-facts-about-the-movie-titanic-1687656865.jpg")

    with overview:
        st.title("Titanic survival prediction")

    with left:
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        embarked_radio = st.radio("Port zaokrętowania", list(embarked_d.keys()), index=2,
                                  format_func=lambda x: embarked_d[x])
        pclass_radio = st.radio("Klasa biletu", list(pclass_d.keys()), index=2, format_func=lambda x: pclass_d[x])

    with right:
        min_age = int(base_data['Age'].min())
        max_age = int(base_data['Age'].max())

        min_sib_sp = int(base_data['SibSp'].min())
        max_sib_sp = int(base_data['SibSp'].max())

        min_parch = int(base_data['Parch'].min())
        max_parch = int(base_data['Parch'].max())

        min_fare = int(base_data['Fare'].min())
        max_fare = int(base_data['Fare'].max())

        age_slider = st.slider("Wiek", value=1, min_value=min_age, max_value=max_age, step=1)
        sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", min_value=min_sib_sp, max_value=max_sib_sp, step=1)
        parch_slider = st.slider("Liczba rodziców i/lub dzieci", min_value=min_parch, max_value=max_parch, step=1)
        fare_slider = st.slider("Cena biletu", min_value=min_fare, max_value=max_fare, step=1)

    data = [[pclass_radio, sex_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]

    column_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    data = pd.DataFrame(data, columns=column_names)
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba przeżyłaby katastrofę?")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == "__main__":
    main()
