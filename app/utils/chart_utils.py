import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_chart(probability):
    plt.figure()
    plt.bar(["No Apto", "Apto"], [1 - probability, probability])
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
