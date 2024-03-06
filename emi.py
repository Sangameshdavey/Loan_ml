from flask import Flask, jsonify, request
import joblib
model = joblib.load('random_forest_model.pkl')
prediction = model.predict([[1.0, 1.0, 2.0, 0.0, 0.0, 2957.0, 0.0, 81.0, 360.0, 1.0, 1.0]])
print(prediction)
app = Flask(__name__)


print("done")
@app.route('/predict', methods=['POST'])
def predict():
    
    # Get the input data from the request
     data = request.json

    # Make prediction using the loaded model
     prediction = model.predict([data['features']])

    # Prepare the response
    
     response = {
        'prediction': int(prediction[0]) # Assuming prediction is a single value
    }
     return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

    
  


#in post man :
# {
#     "features": [1.0, 0.0, 0.0, 0.0, 0.0, 3069.0, 0.0, 71.0, 480.0, 1.0, 2.0]
# }

