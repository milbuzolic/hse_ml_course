import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

st.set_page_config(layout="wide", initial_sidebar_state="auto")

st.title("Расчет стоимости автомобиля")


@st.cache_resource
def load_model_package():
    with open('HW1/model_for_streamlit.pickle', 'rb') as f:
        return pickle.load(f)


data = load_model_package()

car_brands = [
    'Maruti', 'Skoda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
    'Mahindra', 'Honda', 'Chevrolet', 'Fiat', 'Datsun', 'Tata', 'Jeep',
    'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW',
    'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
    'Kia', 'Force', 'Ambassador', 'Isuzu', 'Peugeot'
]

fuel_types = ["Дизель", "Бензин", "Газ (СУГ)", "Газ (КПГ)"]
transsmisions_types = ["Механическая", "Автоматическая"]
seller_types = ["Частное лицо", "Дилер", "Официальный дилер"]
owner_types = ["Первый", "Второй", "Третий",
               "Четвертый и более", "Тест-драйв"]
seats_types = [2, 4, 5, 6, 7, 8, 9, 10, 14]

tab1, tab2, tab3 = st.tabs(["Расчет стоимости", "Визуализация данных", "Веса модели"])

with tab1:
    def prepare_input(user_input, model_data):
        fuel_dict = {
            "Дизель": "Diesel",
            "Бензин": "Petrol",
            'Газ (СУГ)': "LPG",
            'Газ (КПГ)': "CNG"
        }

        transmission_dict = {
            "Механическая": "Manual",
            "Автоматическая": "Automatic"
        }

        seller_dict = {
            "Частное лицо": "Individual",
            "Дилер": "Dealer",
            "Официальный дилер": "Trustmark Dealer"
        }

        owner_dict = {
            "Первый": "First Owner",
            "Второй": "Second Owner",
            "Третий": "Third Owner",
            "Четвертый и более": "Fourth & Above Owner",
            "Тест-драйв": "Test Drive Car"
        }

        user_input_translated = user_input.copy()
        user_input_translated['fuel'] = fuel_dict[user_input['fuel']]
        user_input_translated['transmission'] = transmission_dict[user_input['transmission']]
        user_input_translated['seller_type'] = seller_dict[user_input['seller_type']]
        user_input_translated['owner'] = owner_dict[user_input['owner']]

        processed_features = []

        #Количественные признаки
        for i, feat in enumerate(model_data['numerical_features']):
            value = user_input_translated[feat]
            scaled = (value - model_data['scaler_mean'][i]) / model_data['scaler_std'][i]
            processed_features.append(scaled)

        #Категориальные признаки
        cat_features = model_data['categorical_features']
        encoder_categories = model_data['encoder_categories']

        for i, feat in enumerate(cat_features):
            user_value = user_input_translated[feat]
            categories = encoder_categories[i]

            if user_value in categories:
                idx = list(categories).index(user_value)
                if idx > 0:  # drop first
                    one_hot = [0] * (len(categories) - 1)
                    one_hot[idx - 1] = 1
                    processed_features.extend(one_hot)
            else:
                processed_features.extend([0] * (len(categories) - 1))

        return np.array(processed_features)


    def predict_price(user_input):
        features = prepare_input(user_input, data)
        prediction = data['ridge_intercept']
        for coef, feat in zip(data['ridge_coefficients'], features):
            prediction += coef * feat
        return prediction


    st.subheader("Количественные характеристики")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Год выпуска", value=2018)
        km_driven = st.number_input("Пробег, км", value=50000)
        mileage = st.number_input("Расход топлива, км/л", value=20.0)
        engine = st.number_input("Объем двигателя, куб.см", value=1200)

    with col2:
        max_power = st.number_input("Мощность, л.с.", value=80.0)
        torque = st.number_input("Крутящий момент, Нм", value=160.0)
        max_torque_rpm = st.number_input("Обороты макс. крутящего момента", value=3000)
        seats = st.selectbox("Количество мест", seats_types, index=0)

    st.subheader("Категориальные характеристики")
    cat_col1, cat_col2, cat_col3 = st.columns(3)
    with cat_col1:
        name = st.selectbox("Марка автомобиля", sorted(car_brands))
        fuel = st.selectbox("Тип топлива", fuel_types)
    with cat_col2:
        transmission = st.selectbox("Коробка передач", transsmisions_types)
        seller_type = st.selectbox("Тип продавца", seller_types)
    with cat_col3:
        owner = st.selectbox("Владелец", owner_types)

    if st.button("Рассчитать стоимость"):
        user_input = {
            'year': year,
            'km_driven': km_driven,
            'max_power': max_power,
            'engine': engine,
            'mileage': mileage,
            'torque': torque,
            'max_torque_rpm': max_torque_rpm,
            'seats': seats,
            'name': name,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner
        }
        price = predict_price(user_input)
        st.success(f"### Стоимость автомобиля: {price:,.0f} у.е.".replace(',', ' '))



with tab2:
    st.header("Визуализация данных")

    tab2_1, tab2_2, tab2_3 = st.tabs(["Числовые признаки", "Категориальные признаки", "Марки автомобилей"])

    with tab2_1:
        st.subheader("Зависимость стоимости от числовых признаков")
        with open('HW1/scatter_interactive_selector.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=True)

    with tab2_2:
        st.subheader("Распределение стоимости по категориальным признакам")
        with open('HW1/categorical_boxplots_summary.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=550, scrolling=True)

    with tab2_3:
        st.subheader("Медианная стоимость по маркам автомобилей")
        with open('HW1/brand_median_price_histogram.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=650, scrolling=True)

with tab3:
    st.header("Веса и коэффициенты модели")

    if 'ridge_coefficients' in data:
        st.subheader("Топ-30 важных признаков")

        coef_df = pd.DataFrame({
            'Признак': data['final_feature_order'],
            'Коэффициент': data['ridge_coefficients'],
            'Абс. значение': np.abs(data['ridge_coefficients'])
        })

        top_features = coef_df.sort_values('Абс. значение', ascending=False).head(30)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_features)), top_features['Коэффициент'],color='#0064c8', edgecolor='#0046b4', linewidth=0.5)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Признак'])
        ax.set_xlabel('Значение коэффициента')
        ax.grid(True, alpha=0.3, linestyle='--')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#cccccc')
        ax.tick_params(colors='#333333')

        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        st.pyplot(fig)


        st.subheader("Таблица коэффициентов")
        st.dataframe(coef_df.sort_values('Абс. значение', ascending=False)[['Признак', 'Коэффициент']].style.format({'Коэффициент': '{:.4f}'}))



        st.subheader("Информация о модели")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Intercept", f"{data['ridge_intercept']:,.2f}")
        with col2:
            st.metric("Кол-во коэффициентов", len(data['ridge_coefficients']))

