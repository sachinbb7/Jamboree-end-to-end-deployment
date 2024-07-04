from flask import Flask, render_template , request
import pickle


app = Flask("__name_")


ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
scaler_model = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/form", methods = ["GET", "POST"])
def form():
    if request.method == "POST":
        GRE_score = float(request.form["GRE score"])
        TOEFL_score = float(request.form["TOEFL score"])
        University_rating = float(request.form["University rating"])
        SOP = float(request.form["SOP"])
        LOR= float(request.form["LOR"])
        CGPA = float(request.form["CGPA"])
        Research = float(request.form["Research"])

        real_time_scaled_data = scaler_model.transform([[GRE_score, TOEFL_score, University_rating , SOP, LOR, CGPA , Research]])

        prediction = ridge_model.predict(real_time_scaled_data)
        
        prediction_str = str(prediction[0]*100)

        return render_template("data.html", results = prediction_str)




    else:
        return render_template("data.html")

           


if __name__ == "__main__":
    app.run(debug=True)