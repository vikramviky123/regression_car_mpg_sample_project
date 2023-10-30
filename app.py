from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from train import preProcess

app = Flask(__name__)

candidates = {}


@app.route('/')
def index():
    return render_template('Intro.html')


@app.route('/data')
def data():
    return render_template('data.html')


@app.route('/eda/univariate')
def univariate():
    return render_template('univariate.html')


@app.route('/eda/bivariate')
def bivariate():
    return render_template('bivariate.html')


@app.route('/eda/multivariate')
def multivariate():
    return render_template('multivariate.html')


@app.route('/model/modelanalysis')
def modelanalysis():
    return render_template('modelanalysis.html')


@app.route('/model/modelplots')
def modelplots():
    return render_template('modelplots.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    N_SPLITS = 8
    best_model_name = 'rf_reg'
    pickle_file_path = "models/model_file.pkl"

    # Load the saved list of models using pickle
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_model = pickle.load(pickle_file)

    try:

        # Retrieve the input data from the form
        cyl = float(request.form.get('cyl'))
        disp = float(request.form.get('disp'))
        hp = float(request.form.get('hp'))
        wt = float(request.form.get('wt'))
        acc = float(request.form.get('acc'))
        yr = float(request.form.get('yr'))
        origin = float(request.form.get('origin'))
        car_type = float(request.form.get('car_type'))
        car_name = request.form.get('car_name')

        # Create a dictionary from the form data
        data = {
            'cyl': [cyl],
            'disp': [disp],
            'hp': [hp],
            'wt': [wt],
            'acc': [acc],
            'yr': [yr],
            'origin': [origin],
            'car_type': [car_type],
            'car_name': [car_name]
        }

        # Create a DataFrame from the dictionary
        df = pd.DataFrame(data)
        print(df)

        df_transformed = df.copy()

        # Load the saved list of models using pickle
        pickle_file_path = "models/preprocessing_transformers.pkl"
        with open(pickle_file_path, 'rb') as pickle_file:
            preprocess_pickle = pickle.load(pickle_file)

        process = 'label_encoders'
        print(process)
        for col in preprocess_pickle[process].keys():
            if df_transformed[col].dtype == 'object':
                df_transformed[col] = preprocess_pickle[process][col].transform(
                    df_transformed[col])

        process = 'imputer'
        print(process)
        df_transformed = pd.DataFrame(preprocess_pickle[process].transform(
            df_transformed), columns=df_transformed.columns)

        process = 'scaler'
        print(process)
        df_transformed = pd.DataFrame(preprocess_pickle[process].transform(
            df_transformed), columns=df_transformed.columns)

        print(df_transformed)

        preds_final = np.zeros(1)
        for model in loaded_model:
            preds_final += model.predict(df_transformed) / N_SPLITS

        preds_final = np.round(preds_final, 4)

        return render_template('predict.html', preds_final=preds_final)

    except Exception as e:
        return render_template('predict.html', error_message=str(e))


@app.route('/form')
def show_form():
    return render_template('predict.html', preds_final=None, error_message=None)


if __name__ == '__main__':
    app.run(debug=True)
