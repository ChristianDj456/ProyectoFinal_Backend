import joblib
from app.schemas.candidate_data import CandidateData
from app.utils.chart_utils import generate_chart

# Cargar modelo y escalador
model = joblib.load("app/models/modelo_random_forest.pkl")
scaler = joblib.load("app/models/scaler.pkl")

async def predict(data: CandidateData):
    # Convertir los datos en formato para el modelo

    if (data.age > 35):
        data.age = 1
    elif (data.age < 35):
        data.age = 0
        
    input_data = [[
        data.age, data.accessibility, data.education, data.employment,
        data.gender, data.mental_health, data.main_branch, data.years_code, data.years_code_pro,
        data.salary, data.num_skills, data.continent
    ]]
    input_scaled = scaler.transform(input_data)

    # Realizar predicción
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[:, 1]

    # Generar gráfico
    chart = generate_chart(probability[0])

    return {
        "prediction": "Apto" if prediction[0] == 1 else "No Apto",
        "probability": probability[0],
        "chart": chart
    }
