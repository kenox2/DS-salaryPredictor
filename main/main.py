
import joblib
import pandas as pd

# load dummy dataframe

dummy_df = pd.read_csv("./dummy_df.csv")
print(len(dummy_df.columns))


experience_level = ["Entry", "Mid", "Senior", "Expert"]
company_size = ["S", "M", "L"]
employment_type = ["FT", "CT", "FL", "PT"]
remote_ratio = [0, 50, 100]
job_title = ['Data Engineer', 'Data Scientist', 'Data Analyst',
       'Machine Learning Engineer', 'Analytics Engineer', 'Data Architect',
       'Research Scientist', 'Data Science Manager', 'Applied Scientist',
       'Research Engineer', 'ML Engineer', 'Data Manager',
       'Machine Learning Scientist', 'Data Science Consultant',
       'Data Analytics Manager', 'Computer Vision Engineer', 'AI Scientist',
       'BI Data Analyst', 'Business Data Analyst', 'Data Specialist']

job_title.sort()
prediction_data = []

for ind, level in enumerate(experience_level):
  print(f"{ind+1}. {level}")

expierience = input("input: ")

match expierience:
  case "1":
      prediction_data.append(0)
  case "2":
      prediction_data.append(1)
  case "3":
      prediction_data.append(2)
  case "4":
      prediction_data.append(3)
  case _:
    print("Invalid value chosen. Exiting app....")
    exit()

# get company size
for ind, level in enumerate(company_size):
  print(f"{ind+1}. {level}")

size = input("input: ")
match size:
  case "1":
      prediction_data.append(0)
  case "2":
      prediction_data.append(1)
  case "3":
      prediction_data.append(2)
  case _:
    print("Invalid value chosen. Exiting app....")
    exit()



# get employment type
for ind, level in enumerate(employment_type):
  print(f"{ind+1}. {level}")

expierience = input("input: ")

match expierience:
  case "1":
      prediction_data.append(0)
  case "2":
      prediction_data.append(1)
  case "3":
      prediction_data.append(2)
  case "4":
      prediction_data.append(3)
  case _:
    print("Invalid value chosen. Exiting app....")
    exit()


# get remote ratio
for ind, level in enumerate(remote_ratio):
  print(f"{ind+1}. {level}")

size = input("input: ")
match size:
  case "1":
      prediction_data.append(0)
  case "2":
      prediction_data.append(50)
  case "3":
      prediction_data.append(100)
  case _:
    print("Invalid value chosen. Exiting app....")
    exit()

# get job
for ind, level in enumerate(job_title):
  print(f"{ind+1}. {level}")

job = input("input: ")

if int(job) not in range(1,20):
  print("Invalid value chosen. Exiting app....")
  exit()

for i in range(20):
  if((int(job)-1)==i):
    prediction_data.append(1)
  else:
    prediction_data.append(0)


dummy_df.loc[len(dummy_df.index)] = prediction_data



# Load model
loaded_rf = joblib.load("./random_forest.joblib")


# Standarize values and make pred

row_to_scale = [dummy_df.iloc[0]]

# Load scaler
scaler = joblib.load('scaler.pkl')

# Scale the features
scaled_row = scaler.transform(row_to_scale)

# Make predictions using your trained model
prediction = loaded_rf.predict(scaled_row)

print("Prediction:", prediction)

dummy_df = dummy_df.iloc[0:0]

dummy_df.to_csv("./dummy_df.csv")
