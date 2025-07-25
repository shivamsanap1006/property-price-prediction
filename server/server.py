from flask import Flask, request, jsonify
import util
import numpy as np

app = Flask(__name__)
app.debug = True


@app.route("/get_location_names", methods=["GET"])
def get_location_names():
    response = jsonify({"locations": util.get_location_names()})
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response

@app.route("/get_location_names_rent", methods=["GET"])
def get_location_names_rent():
    response = jsonify({"locations": util.get_rent_locations()})
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


@app.route("/predict_home_price", methods=["GET", "POST"])
def predict_home_price():
    total_sqft = float(request.form["total_sqft"])
    location = request.form["location"]
    bhk = int(request.form["bhk"])
    bath = int(request.form["bath"])

    response = jsonify(
        {"estimated_price": util.get_estimated_price(location, total_sqft, bhk, bath)}
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/rent_price", methods=["GET", "POST"])
def predict_rent():
    location = request.form["location"]
    bhk = int(request.form["bhk"])
    bathrooms = int(request.form["bath"])
    print("Location : " , location , "BHK : " , bhk , "Bath : " , bathrooms)
    predict_rent_price = util.get_predicted_rent(location , bhk , bathrooms)
    # predict_rent_price = util.get_predicted_rent('1st Phase JP Nagar',2,2)
    print(predict_rent_price)
    response = jsonify(
        {"estimated_price": predict_rent_price}
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    util.load_rent_artifacts()
    app.run(debug=True)
