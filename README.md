# House Price Prediction

## Overview
Predicting residential house prices using Linear Regression
based on area income, house age, rooms, bedrooms and population.

## Dataset
- Source: Kaggle (House_price.csv)
- 4448 samples, 5 features

## What I Did
1. Dropped irrelevant columns (Address)
2. Split data into training and testing sets
3. Scaled features using StandardScaler
4. Trained Linear Regression model
5. Evaluated model using R² score
6. Built an interactive input system to predict custom house prices

## Code Explanation ( start from Line 33 to 49 )

### Ranges
Getting the min and max of each feature directly from the dataset.
This ensures the user enters realistic values that the model was trained on,
not random numbers that could give wrong predictions.

### Input Function (get_input)
A function that asks the user to enter a value for each feature.
It keeps asking until the user enters a valid value within the dataset range.

Example:
- If Income in the dataset is between 17,796 and 107,701
- And you enter 500 → it will reject it and ask again ( It will tell you the allowed limits )
- If you enter 50,000 → it accepts it and moves on

Why float instead of int?
- float is safer for ML inputs
- int loses decimal precision (e.g. 50000.5 becomes 50000)
- Some features like smoothness need decimals to be accurate

## Results
| Metric | Score |
|--------|-------|
| Accuracy (Test) | ~91% |

## Libraries
- Python
- Pandas
- NumPy
- Scikit-Learn
## How to Run
1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Run: python house-price.py
4. Enter the required inputs when prompted
5. Don't forget to change the file path!
