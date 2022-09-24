import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('cancer_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [eval(x) for x in request.form.values()]

    # int_features = [np.array(int_features)]
    # sc = StandardScaler()
    # int_features = sc.fit_transform(int_features)
    prediction = model.predict([int_features])
    prediction = (prediction>0.5)
    print(prediction)
    var = ""
    if prediction[0] == True:
        var = "You have breast cancer"
    elif prediction[0] == False:
        var = "Hurrah! your body looks good"
    return render_template('index.html', prediction_text='{}'.format(var))

@app.route('/cancerpredict_api',methods=['POST'])
def cancerpredict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)