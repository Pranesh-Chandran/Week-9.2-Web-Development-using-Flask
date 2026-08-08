from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model
model = pickle.load(open("models/SOH_Final_Model.sav", "rb"))

#Load scaler
scaler = pickle.load(open("models/scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    Cycles = float(request.form["Cycles"])
    Voltage_V = float(request.form["Voltage_V"])
    Current_A = float(request.form["Current_A"])
    Temperature_C = float(request.form["Temperature_C"])
    Internal_Resistance_mOhm = float(request.form["Internal_Resistance_mOhm"])
    Capacity_Ah = float(request.form["Capacity_Ah"])

    #data = np.array([[bgr, bu, sc, pcv, wc]])
    #data = np.array([[Cycles, Voltage_V, Current_A, Temperature_C, Internal_Resistance_mOhm, Capacity_Ah]])

    data = pd.DataFrame(
    np.array([[Cycles, Voltage_V, Current_A, Temperature_C, Internal_Resistance_mOhm, Capacity_Ah]]),
    columns=["Cycles", "Voltage_V", "Current_A", "Temperature_C", "Internal_Resistance_mOhm", "Capacity_Ah"])

    print("data:", data)
    scaled_data = scaler.transform(data)
    print("scaled_data:", scaled_data)
    
    prediction = model.predict(scaled_data)[0]
    print("Prediction:", prediction)

    # Keep prediction within battery health range
    #prediction = max(0, min(100, prediction))
    #prediction = round(prediction, 2)

    prediction = round(float(prediction), 2)

    if prediction >= 80:
        result = f"🟢 Healthy Battery ({prediction}%)"
    elif prediction >= 50:
        result = f"🟡 Moderate Battery Health ({prediction}%)"
    else:
        result = f"🔴 Poor Battery Health ({prediction}%)"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)