# Adjusted Price Prediction

This project predicts the adjusted price of menu items for a restaurant based on various factors, including the item's base price, the lowest competitive price, weather conditions, time of the day, and whether the restaurant is busy. The model uses machine learning techniques to make predictions and dynamically fetches real-time weather data using the OpenWeatherMap API.

---

## **Project Highlights**

1. **Machine Learning Model**: A Random Forest Regressor tuned using GridSearchCV for optimal hyperparameters.
2. **Data Preprocessing**: Label encoding for categorical variables and standard scaling for numerical features.
3. **Real-time Inputs**: Fetches current weather conditions and temperature using OpenWeatherMap API.
4. **Custom Business Logic**: Includes shop closure conditions (e.g., before 11:30 AM or all day on Tuesdays).
5. **Evaluation Metrics**: Model evaluated using Mean Squared Error (MSE), R² Score, and Mean Absolute Error (MAE).

---

## **Features**

- **Dynamic Weather Integration**: Automatically retrieves live weather data for predictions.
- **Business Hours Logic**: Predicts prices only during business hours.
- **Detailed Predictions**: Provides adjusted price predictions for all items in the menu.

---

## **Technologies Used**

- **Python Libraries**: Pandas, NumPy, Scikit-learn, Requests, Flask/Streamlit.
- **Modeling**: Random Forest Regressor.
- **API Integration**: OpenWeatherMap API.

---

## **Dataset**
The dataset was generated using web scraping techniques, with a dedicated Google Colab notebook created and uploaded to this repository for reproducibility. The notebook demonstrates step-by-step how the data was extracted from yelp and prepared for the prediction model.


- **Input**: CSV file (`data.csv`) containing menu items with the following fields:

  - `Item`: Name of the menu item.
  - `price`: Base price of the item.
  - `lowest competitive price`: Lowest competitor price.
  - `Day`: Day of the week.
  - `Hour`: Time of the day.
  - `Temperature`:current temperature
  - `Is busy`: 1 if the restaurant is busy, else 0.
  - `weather`: Current weather condition.

- **Output**: Adjusted price predictions for menu items based on the above features.

---

## **Model Evaluation Metrics**

The model was evaluated using:

1. **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
   - Example: `MSE = 1.91`
2. **R² Score**: Explains the proportion of variance captured by the model.
   - Example: `R² Score = 0.86`
3. **Mean Absolute Error (MAE)**: Calculates the average absolute difference between predictions and actual values.
   - Example: `MAE = 0.96`

---

## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone https://github.com/somewhereverywhere/Price_Prediction.git
cd Price_prediction
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Add OpenWeather API Key**

Replace `YOUR_API_KEY` in the code with your OpenWeather API key.

### **4. Run the Script**

```bash
python main.py
```

---

## **How to Run**

### **Step-by-Step Execution**

1. **Load and Preprocess Data**:

   - The dataset is read from a CSV file (`final2.csv`).
   - Features like `weather` and `Day` are encoded using LabelEncoder.
   - Numerical features (`Temperature` and `Hour`) are normalized using StandardScaler.

2. **Train the Model**:

   - Random Forest Regressor is trained and tuned using GridSearchCV.

3. **Real-time Weather Data**:

   - Temperature and weather conditions are fetched from OpenWeatherMap API.

4. **Make Predictions**:

   - Adjusted prices are predicted for each menu item.

5. **Display Results**:

   - Outputs displayed either in the console or on a webpage.

---

## **Output Display**

### **Console Outputs**

- Display real-time weather and temperature:

```plaintext
Current Temperature: 72.5°F
Current Weather: Clear
```

- Predicted prices:

```plaintext
Predictions:
Item: Burger, Predicted Adjusted Price: 12.34
Item: Pizza, Predicted Adjusted Price: 15.67
```

### **Webpage Outputs (Optional)**

You can use Flask or Streamlit for a user-friendly webpage interface:

- **Flask**: Create an HTML interface to display predictions dynamically.
- **Streamlit**: Build an interactive dashboard to show the predictions and real-time weather data.

---

## **Files in Repository**

- `main.py`: Main script for prediction.
- `data.csv`: Input dataset.
- `final2_processed`: input data after processing.
- `data.py`:script for data preprocessing and model training.
- `adjusted_price_model.pkl`: Pre-trained model.
- `scaler.pkl`: StandardScaler object for numerical features.
- `label_encoder_weather.pkl`: LabelEncoder for weather.
- `label_encoder_day.pkl`: LabelEncoder for days.
- `README.md`: Documentation.

---

## **Sample Outputs**

### **Real-time Weather Data**

```plaintext
Current Temperature: 65.5°F
Current Weather: Rainy
```

### **Predicted Adjusted Prices**

```plaintext
Item: Aloo, Predicted Adjusted Price: $9.45
```

---

## **Future Enhancements**

1. **Include Additional Features**:
   - Customer preferences.
   - Holiday data.
2. **Improve UI**:
   - Add a graphical dashboard for enhanced visualization.
3. **Deploy the Model**:
   - Use platforms like AWS, Heroku, or Google Cloud.

---

## **License**

This project is licensed under the MIT License. See `LICENSE` for more information.

---

For questions or suggestions, feel free to reach out!




