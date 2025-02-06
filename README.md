# README.md

# H2 Percentage Prediction App

This project is a Streamlit application that utilizes a machine learning model to make predictions based on user-uploaded Excel files. The app features a modern interface and includes functionality for feature extraction, fuzzy matching, and result display.

## Project Structure

```
├── models
│   └── model.joblib        # Your pre-trained model file
├── app.py                  # The main Streamlit app
├── requirements.txt        # List of dependencies
└── README.md               # This file

```


## Features

- Upload an Excel file with multiple features.
- Extract the required 50 features using fuzzy matching.
- Select from multiple matches via a dropdown menu.
- Predict the target variable using the loaded machine learning model.
- Display the prediction results.
- Download the modified Excel file with the predicted target added as a new column.

## License

This project is licensed under the MIT License.