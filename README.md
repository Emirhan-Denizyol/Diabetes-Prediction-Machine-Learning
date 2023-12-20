I developed a model that predicts whether people have diabetes or not based on the information given in this data set. For this model I used KNN, CART, RF, XGBoost, LightGBM, CATBoost. Among these models, I chose the most successful model with VotingClassifier.

Context
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.


1- Pregnancies: Number of times pregnant
2-Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3- BloodPressure: Diastolic blood pressure (mm Hg)
4- SkinThickness: Triceps skin fold thickness (mm)
5- Insulin: 2-Hour serum insulin (mu U/ml)
6- BMI: Body mass index (weight in kg/(height in m)^2)
7- DiabetesPedigreeFunction: Diabetes pedigree function
8- Age: Age (years)
9- Outcome: Class variable (0 or 1)

