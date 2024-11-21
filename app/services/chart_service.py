import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px


def load_data():
    return pd.read_csv("app/models/stackoverflow_full.csv")

def segment_country(country):
    if country in ['United States of America', 'Canada', 'Mexico']:
        return 'North America'
    elif country in ['United Kingdom', 'France', 'Germany', 'Spain', 'Italy']:
        return 'Europe'
    elif country in ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru']:
        return 'South America'
    elif country in ['China', 'Japan', 'India', 'South Korea']:
        return 'Asia'
    elif country in ['Australia', 'New Zealand']:
        return 'Australia'
    else:
        return 'Other'

# Crear una copia de df y aplicar Label Encoding
def create_encoded_df():
    df = load_data()
    df_copy = df.copy()
    
    # Crear la columna 'Continent' si no existe
    if 'Continent' not in df_copy.columns:
        df_copy['Continent'] = df_copy['Country'].apply(segment_country)
    
    # Aplicar Label Encoding a las columnas categóricas
    label_encoder = LabelEncoder()
    categorical_columns = ['Age', 'Accessibility', 'EdLevel', 'Gender', 'MentalHealth', 'MainBranch', 'Continent']
    for col in categorical_columns:
        df_copy[col] = label_encoder.fit_transform(df_copy[col])
    
    return df_copy

# Generar la vista preliminar de df_copy (primeras filas)
def generate_encoded_head_table():
    df_copy = create_encoded_df()
    head_data = df_copy.head().to_dict(orient="records")  # Convertir las primeras 5 filas a formato JSON
    return head_data

def generate_correlation_heatmap():
    df_copy = create_encoded_df()
    
    # Seleccionar solo columnas numéricas para la correlación
    numeric_df = df_copy.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_df.corr()

    # Ajustar el tamaño de la figura
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, cmap="coolwarm", fmt=".2f")
    plt.title('Mapa de Calor de la Correlación')

    # Guardar la imagen como Base64
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")  # bbox_inches="tight" ayuda a ajustar los bordes
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# Generar el resumen estadístico de df_copy
def generate_encoded_describe_table():
    df_copy = create_encoded_df()
    describe_data = df_copy.describe().to_dict()  # Convertir el resumen estadístico a formato JSON
    return describe_data


def generate_education_level_chart():
    df = load_data()
    

    education_counts = df['EdLevel'].value_counts().reset_index()
    education_counts.columns = ['EdLevel', 'Count']

    # Crear un gráfico de barras horizontales con más espacio horizontal
    fig = px.bar(education_counts, y='EdLevel', x='Count', orientation='h', title='Candidatos por Niveles de Educación')
    fig.update_layout(
        xaxis_title='Candidatos',
        yaxis_title='Nivel de Educación',

    )
    
    return fig.to_json()

# Función para generar el gráfico de distribución por edades
def generate_age_distribution_chart():
    df = load_data()
    fig = px.histogram(df, x='Age')
    fig.update_layout(xaxis_title='Edades', yaxis_title='Candidatos')
    
    return fig.to_json()

def segment_country(country):
    if country in ['United States of America', 'Canada', 'Mexico']:
        return 'América del Norte'
    elif country in ['United Kingdom of Great Britain and Northern Ireland', 'France', 'Germany', 'Spain', 'Italy', 'Portugal', 'Belgium', 'Netherlands', 'Austria', 'Switzerland', 'Denmark', 'Ireland', 'Norway', 'Sweden', 'Finland', 'Greece', 'Czech Republic', 'Slovakia', 'Hungary', 'Poland']:
        return 'Europa'
    elif country in ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela, Bolivarian Republic of...', 'Bolivia']:
        return 'América del Sur'
    elif country in ['China', 'Japan', 'South Korea', 'Viet Nam', 'India', 'Sri Lanka', 'Pakistan', 'Bangladesh', 'Indonesia', 'Malaysia', 'Philippines', 'Taiwan', 'Thailand', 'Cambodia', 'Myanmar', 'Laos', 'Singapore', 'Hong Kong (S.A.R.)']:
        return 'Asia'
    elif country in ['Australia', 'New Zealand', 'Fiji', 'Papua New Guinea', 'Solomon Islands', 'Vanuatu', 'Samoa', 'Tonga']:
        return 'Oceanía'
    else:
        return 'Otros'

def generate_continent_distribution_chart():
    df = load_data()
    df['Continent'] = df['Country'].apply(segment_country)
    continent_counts = df['Continent'].value_counts().reset_index()
    continent_counts.columns = ['Continent', 'Count']

    fig = px.bar(continent_counts, x='Continent', y='Count')
    fig.update_layout(xaxis_title='Continente', yaxis_title='Candidatos')
    
    return fig.to_json()

# Función para generar el gráfico de relación entre edad y estado de empleo
def generate_age_employment_chart():
    df = load_data()
    fig = px.histogram(df, x='Age', color='Employment', barmode='group')
    fig.update_layout(xaxis_title='Edad', yaxis_title='Conteo')
    
    return fig.to_json()

def generate_continent_pie_chart():
    df = load_data()
    df['Continent'] = df['Country'].apply(segment_country)
    continent_counts = df['Continent'].value_counts().reset_index()
    continent_counts.columns = ['Continent', 'Count']

    fig = px.pie(continent_counts, names='Continent', values='Count')
    
    # Actualizar trazos para hacer el texto de las etiquetas blanco y más visible
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        marker=dict(line=dict(color='black', width=1)),  # Bordes para separar las secciones
        textfont=dict(color='white')  # Cambiar el color de la fuente a blanco
    )
    fig.update_layout(

        legend=dict(
            font=dict(size=10),
            orientation="v",
            x=1,
            y=1
        )
    )
    
    return fig.to_json()

def generate_computer_skills_heatmap():
    df = load_data()
    df['Continent'] = df['Country'].apply(segment_country)  # Asegúrate de que el continente esté en el DataFrame

    # Crear una tabla cruzada para EdLevel (nivel de educación) y Continent, con la frecuencia de ComputerSkills
    heatmap_data = pd.crosstab(df['EdLevel'], df['Continent'], values=df['ComputerSkills'], aggfunc='mean').fillna(0)

    # Crear el mapa de calor con Plotly, usando la paleta 'Blues' para representar los niveles de ComputerSkills
    fig = px.imshow(heatmap_data, text_auto=True, color_continuous_scale="Blues",
                    title="Habilidades de Computación")
    fig.update_layout(
        xaxis_title="Continente",
        yaxis_title="Nivel de Educación",
        width=600,  # Ajustar el ancho
        height=500  # Ajustar la altura
    )
    
    return fig.to_json()


# Función para generar la tabla de conteo de la variable 'Employed'
def generate_employed_counts():
    df_copy = create_encoded_df()
    employed_counts = df_copy['Employed'].value_counts().to_dict()
    return employed_counts

# Función para generar los boxplots de 'YearsCode', 'PreviousSalary' y 'ComputerSkills'
def generate_boxplot_distributions():
    df_copy = create_encoded_df()
    
    # Crear la figura con los tres subplots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    sns.boxplot(data=df_copy, y='YearsCode', orient='vertical')
    plt.title('Años de Experiencia')
    
    plt.subplot(132)
    sns.boxplot(data=df_copy, y='PreviousSalary', orient='vertical')
    plt.title('Salario Previo')
    
    plt.subplot(133)
    sns.boxplot(data=df_copy, y='ComputerSkills', orient='vertical')
    plt.title('Habilidades Computacionales')
    
    # Convertir la figura a una imagen en Base64
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# Define a function to remove outliers using IQR
def remove_outliers_iqr(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    return data

# Remove outliers in 'YearsCode', 'PreviousSalary,' and 'ComputerSkills'
# Función para generar la descripción estadística de X
def generate_x_description():
    df_copy = create_encoded_df()
    
    # Aplicar eliminación de valores atípicos en las columnas relevantes
    df_copy = remove_outliers_iqr(df_copy, 'YearsCode')
    df_copy = remove_outliers_iqr(df_copy, 'PreviousSalary')
    df_copy = remove_outliers_iqr(df_copy, 'ComputerSkills')

    X = df_copy.drop(columns=['Employed'])
    x_description = X.describe().to_dict()
    return x_description

# Función para generar la descripción estadística de y
def generate_y_description():
    df_copy = create_encoded_df()
    
    # Aplicar eliminación de valores atípicos en las columnas relevantes
    df_copy = remove_outliers_iqr(df_copy, 'YearsCode')
    df_copy = remove_outliers_iqr(df_copy, 'PreviousSalary')
    df_copy = remove_outliers_iqr(df_copy, 'ComputerSkills')

    y = df_copy['Employed']
    y_description = y.describe().to_dict()
    return y_description

# Función para generar la curva ROC
# Función para preparar los datos y entrenar el modelo de Random Forest
def train_random_forest():
    df_copy = create_encoded_df()

    # Remove outliers
    df_copy = remove_outliers_iqr(df_copy, 'YearsCode')
    df_copy = remove_outliers_iqr(df_copy, 'PreviousSalary')
    df_copy = remove_outliers_iqr(df_copy, 'ComputerSkills')

    # Separate X and y
    X = df_copy.drop(columns=['Employed'])
    y = df_copy['Employed']

    # Select only numeric columns
    X_numeric = X.select_dtypes(include=[np.number])

    # Normalize the numeric columns of X
    scaler = StandardScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )

    # Train the Random Forest model
    model = RandomForestClassifier(
        random_state=42, max_depth=7, min_samples_split=3, n_estimators=50
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

# Función para generar la gráfica de curva ROC
def generate_roc_curve():
    model, X_train, X_test, y_train, y_test = train_random_forest()
    y_scores_rf = model.predict_proba(X_train)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_train, y_scores_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure()
    plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Curva ROC (área = {auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Random Forest - Curva ROC')
    plt.legend(loc='lower right')

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# Función para generar la matriz de confusión
def generate_confusion_matrix():
    model, _, X_test, _, y_test = train_random_forest()
    y_pred = model.predict(X_test)

    cm_rf = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Employed', 'Employed'], yticklabels=['Not Employed', 'Employed'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Random Forest - Matriz de Confusión')

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"