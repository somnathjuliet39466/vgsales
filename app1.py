import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model1.pkl", "rb"))


@flask_app.route("/")
def home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    # Get the form values
    platform = request.form['Platform']
    genre = request.form['Genre']
    na_sales = float(request.form['NA_Sales'])
    eu_sales = float(request.form['EU_Sales'])
    jp_sales = float(request.form['JP_Sales'])
    other_sales = float(request.form['Other_Sales'])

    # Create a DataFrame with the input values
    input_data = pd.DataFrame([[platform, genre, na_sales, eu_sales, jp_sales, other_sales]],
                              columns=['Platform', 'Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])

    # Load the saved model
    model = pickle.load(open("model1.pkl", "rb"))

    # Perform prediction
    prediction = model.predict(input_data)

    # Extract the scalar value from the NumPy ndarray
    scalar_prediction = prediction.item()

    # return render_template("index.html", prediction_text="Predicted information of Global sales: {}".format(prediction))
    formatted_prediction = "{:.2f}".format(scalar_prediction)

    return render_template("index.html",
                           prediction_text="Predicted information of Global sales: {}%".format(formatted_prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)
