from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.candidate_data import CandidateData
from app.services.predict_service import predict
import pandas as pd
import io
from typing import List
from typing import Dict, Any
# from app.utils.chart_utils import generate_dashboard_chart
from app.services.chart_service import (
    generate_age_distribution_chart,
    generate_age_employment_chart,
    generate_boxplot_distributions,
    generate_computer_skills_heatmap,
    generate_confusion_matrix,
    generate_continent_distribution_chart,
    generate_continent_pie_chart,
    generate_correlation_heatmap,
    generate_education_level_chart,
    generate_employed_counts,
    generate_encoded_describe_table,
    generate_encoded_head_table,
    generate_roc_curve,
    generate_x_description,
    generate_y_description,
    load_data
)

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.get("/")
def read_root():
    return {"message": "¡Backend desplegado correctamente!"}

@app.post("/predict")
async def predict_candidate(data: CandidateData):
    return await predict(data)

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
    # Leer el contenido del archivo
    contents = await file.read()
    
    # Decodificar y cargar en un DataFrame
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    df.drop(columns=['Employed'])
    # Lista para almacenar los resultados de las predicciones
    predictions = []

    # Iterar sobre cada fila del DataFrame y realizar la predicción
    for _, row in df.iterrows():
        try:
            # Construir un objeto CandidateData con los datos de la fila
            candidate = CandidateData(
                age=row['Age'],
                accessibility=row['Accessibility'],
                education=row['EdLevel'],
                employment=row['Employment'],
                gender=row['Gender'],
                mental_health=row['MentalHealth'],
                main_branch=row['MainBranch'],
                years_code=row['YearsCode'],
                years_code_pro=row['YearsCodePro'],
                salary=row['PreviousSalary'],
                num_skills=row['ComputerSkills'],
                continent=row['Continent']
            )

            # Realizar la predicción para el candidato
            prediction_result = await predict(candidate)

            # Agregar el resultado con los datos del candidato y la probabilidad a la lista
            predictions.append({
                "name": row.get('Name', ''),
                 "age": candidate.age,
                 "gender": candidate.gender,
                 "education": candidate.education,
                "probability": prediction_result["probability"]  # 'probability' como campo devuelto
            })

        except Exception as e:
            print(f"Error en el procesamiento de la fila {row}: {e}")

    # Devolver la lista de predicciones como respuesta
    print(predictions)
    return predictions


@app.get("/dashboard")
async def dashboard():
    # Obtener las tablas de df y df_copy
    table_data = load_data().head().to_dict(orient="records")
    describe_data = load_data().describe().to_dict()
    encoded_head_data = generate_encoded_head_table()
    encoded_describe_data = generate_encoded_describe_table()

    return {
        "charts": {
            "education_level": generate_education_level_chart(),
            "age_distribution": generate_age_distribution_chart(),
            "age_employment": generate_age_employment_chart(),
            "continent_distribution": generate_continent_distribution_chart(),
            "continent_pie": generate_continent_pie_chart(),
            "computer_skills_heatmap": generate_computer_skills_heatmap(),
            "correlation_heatmap": generate_correlation_heatmap(),  # Agregar el mapa de calor
            "employed_counts": generate_employed_counts(),
            "boxplot_distributions": generate_boxplot_distributions(),
            "roc_curve": generate_roc_curve(),
            "confusion_matrix": generate_confusion_matrix(),
            "roc_curve": generate_roc_curve(),
            "confusion_matrix": generate_confusion_matrix(),

        },
        "tables": {
            "x_description": generate_x_description(),
            "y_description": generate_y_description(),
        },
        "table_data": table_data,
        "describe_data": describe_data,
        "encoded_head_data": encoded_head_data,  # Agregar vista preliminar de df_copy
        "encoded_describe_data": encoded_describe_data , # Agregar resumen estadístico de df_copy
        
    }