import pandas as pd


def create_recursive_data(data: pd.DataFrame, window_size: int = 5, targets: int = 2)-> pd.DataFrame:
   # Make features 
   i = 1
   while i < window_size:
      data[f'MinTemp_{i}'] = data['MinTemp'].shift(-i)
      data[f'MaxTemp_{i}'] = data['MaxTemp'].shift(-i)
      data[f'Rainfall_{i}'] = data['Rainfall'].shift(-i)
      i+=1
      
   # Make targets
   i = 0
   while i < targets:
      data[f'MinTemp_target_{i}'] = data['MinTemp'].shift(-window_size-i)
      data[f'MaxTemp_target_{i}'] = data['MaxTemp'].shift(-window_size-i)
      data[f'Rainfall_target_{i}'] = data['Rainfall'].shift(-window_size-i)
      i+=1
   
   # Clear missing values (any last rows)
   data = data.dropna(axis=0)
   return data
   

data = pd.read_csv('weatherAUS.csv')
print(data.info())

data = data[['MinTemp', 'MaxTemp', 'Rainfall']] # Filter cols: [Date, MinTemp, MaxTemp, Rainfall]

# Fill null values with mean of the column
data['MinTemp'] = data['MinTemp'].fillna(data['MinTemp'].mean())
data['MaxTemp'] = data['MaxTemp'].fillna(data['MaxTemp'].mean())
data['Rainfall'] = data['Rainfall'].fillna(data['Rainfall'].mean())

print(data.info())
data = create_recursive_data(data)
# data.to_csv('processed_weatherAUS.csv', index=False)
