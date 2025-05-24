import joblib
model = joblib.load("tuning/artifacts/knn_v4.pkl") # best model 
print("model loaded successfully")
print(model)

features = [[1.1, 2.1, 1.4, 2.2, 9.4, 1.7, 6.4, 7.0, 2.0]] # Sample, (1, 9)
prediction = model.predict(features)
print(f"output - {prediction}")