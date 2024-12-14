import pandas as pd
import pickle
import requests
from datetime import datetime

# Function to get the weather and temperature for New York using OpenWeatherMap
def get_weather_data():
    api_key = "YOUR_API_KEY"
    city = "New York"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=imperial"  # Imperial for Fahrenheit
    response = requests.get(url)
    data = response.json()
    temperature = data["main"]["temp"]
    weather_condition = data["weather"][0]["main"]
    return temperature, weather_condition

# Function to determine if the restaurant is busy based on time (6-9 pm)
def is_busy():
    current_hour = datetime.now().hour
    # Restaurant is busy between 6 PM (18:00) and 9 PM (21:00)
    return 1 if 18 <= current_hour <= 21 else 0

# Get today's day
def get_day():
    current_day = datetime.now().strftime('%A')  # Get the day as a string
    if current_day == 'Tuesday':
        print("The shop is closed on Tuesdays.")
        return None  # Return None to indicate the shop is closed
    return current_day
def get_hour():
    current_time = datetime.now()
    if current_time.hour < 11 or (current_time.hour == 11 and current_time.minute < 30):
        print("The shop is currently closed. It opens at 11:30 AM.")
        return None  # Return None to indicate the shop is closed
    return current_time.hour

# Load the dataset
df = pd.read_csv("final2_processed.csv")

# Prepare a list to store predictions
predictions = []

# Get weather, temperature, is_busy, day, and hour
temperature_fahrenheit, weather_condition = get_weather_data()
busy_status = is_busy()
day = get_day()
time = get_hour()

# Load the saved LabelEncoder and StandardScaler used during training
with open("label_encoder_weather.pkl", "rb") as le_weather_file, \
     open("scaler.pkl", "rb") as scaler_file, \
     open("label_encoder_day.pkl", "rb") as le_day_file:
    le_weather = pickle.load(le_weather_file)
    scaler = pickle.load(scaler_file)
    le_day = pickle.load(le_day_file)

# Loop over all items in the dataset
for idx, row in df.iterrows():
    # Get the item data for each row
    item = row["Item"]
    price = row["price"]
    lowest = row["lowest competitive price"]

    # Prepare input data for prediction
    input_data = {
        "Item": item,
        "price": price,
        "lowest competitive price": lowest,
        "Day": day,
        "Hour": time,
        "Temperature": temperature_fahrenheit,
        "weather": weather_condition,
        "Is busy": busy_status,
    }

    input_data_df = pd.DataFrame([input_data])

    # Encode categorical variables
    input_data_df["weather"] = le_weather.transform([input_data_df["weather"].iloc[0]])[0]
    input_data_df["Day"] = le_day.transform([input_data_df["Day"].iloc[0]])[0]

    # Apply scaling on the numerical features (Temperature, Hour)
    input_data_df[["Temperature", "Hour"]] = scaler.transform(input_data_df[["Temperature", "Hour"]])

    # Load the trained model
    with open("adjusted_price_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    # Predict the adjusted price using the model
    predicted_price = model.predict(input_data_df.drop(columns=["Item"]))

    # Add the prediction to the list
    predictions.append({
        "Item": item,
        "Predicted Adjusted Price": round(predicted_price[0], 2)  # Round to two decimal places
    })

# Convert the predictions list to a DataFrame
predictions_df = pd.DataFrame(predictions)

# Merge with the original dataset for reference
result_df = pd.merge(df, predictions_df, on="Item", how="left")

# Display the result with predictions
print(f"Current Temperature: {temperature_fahrenheit}°F")
print(f"Current Weather Condition: {weather_condition}")
print(result_df[["Item", "Predicted Adjusted Price"]])

# Optionally, save the result to a CSV file
result_df.to_csv("predicted_prices.csv", index=False)

