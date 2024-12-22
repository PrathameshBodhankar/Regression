# Black Friday Sales Regression Analysis

## Project Overview
This project analyzes the Black Friday Sales dataset to predict purchase amounts using regression models. The dataset includes demographic and transactional data, enabling us to explore patterns and build predictive models to better understand factors influencing customer spending during Black Friday sales.

## Dataset
The dataset contains customer demographics, product details, and purchase information.

### Key Features:
- **User_ID**: Unique identifier for customers.
- **Product_ID**: Unique identifier for products.
- **Gender**: Gender of the customer (Male/Female).
- **Age**: Age group of the customer.
- **Occupation**: Occupation code of the customer.
- **City_Category**: Category of the city (A/B/C).
- **Stay_In_Current_City_Years**: Number of years the customer has stayed in the current city.
- **Marital_Status**: Marital status of the customer (0 = Unmarried, 1 = Married).
- **Product_Category_1, 2, 3**: Categories of purchased products.
- **Purchase**: Amount spent by the customer (Target variable).

## Objectives
1. Perform exploratory data analysis (EDA) to identify key patterns and insights.
2. Preprocess the dataset by handling missing values, encoding categorical variables, and scaling features.
3. Build and evaluate regression models to predict purchase amounts.
4. Interpret the results to provide actionable business insights.

## Installation
### Prerequisites
- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - jupyterlab (optional, for interactive exploration)

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/black-friday-sales-regression.git
   ```
2. Navigate to the project directory:
   ```bash
   cd black-friday-sales-regression
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Project Workflow
### 1. Data Exploration
- Analyze the distribution of the target variable (`Purchase`).
- Explore relationships between customer demographics and purchase amounts.
- Identify correlations between features.

### 2. Data Preprocessing
- **Handle Missing Values**: Impute missing values in product categories.
- **Encode Categorical Variables**: Use label encoding and one-hot encoding for features like `Gender`, `Age`, and `City_Category`.
- **Feature Scaling**: Standardize numerical features to improve model performance.
- **Feature Engineering**: Create new features (e.g., total products purchased).

### 3. Model Building
- Train multiple regression models, including:
  - Linear Regression
  - Ridge and Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Split data into training and test sets using an 80-20 split.

### 4. Model Evaluation
- Evaluate models using:
  - **R-squared**: Measures the proportion of variance explained by the model.
  - **Mean Absolute Error (MAE)**: Average absolute errors between predicted and actual values.
  - **Mean Squared Error (MSE)**: Average squared errors between predicted and actual values.

### 5. Results and Insights
- Identify key features impacting purchase amounts.
- Compare model performance and select the best-performing model.

## Example Code
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('BlackFriday.csv')

# Preprocessing steps (e.g., handle missing values, encode categorical variables)
# ...

# Split data
X = data.drop('Purchase', axis=1)
y = data['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"MSE: {mse}, R-squared: {r2}")
```

## Visualizations
- Histograms and box plots for `Purchase` distribution.
- Bar charts to compare purchase behavior across age groups and city categories.
- Feature importance plots for tree-based models.

## File Structure
```
├── data
│   ├── BlackFriday.csv             # Raw dataset
├── notebooks
│   ├── data_exploration.ipynb      # EDA and preprocessing
│   ├── model_building.ipynb        # Regression model training
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
```

## Insights
1. Younger customers (age group 18-25) tend to spend more on Black Friday.
2. Married customers show a higher average purchase amount compared to unmarried customers.
3. City category plays a significant role in purchase behavior, with Category A cities leading in average spending.

## Future Work
- Add advanced regression techniques like XGBoost and LightGBM.
- Perform hyperparameter tuning for model optimization.
- Explore customer segmentation for personalized marketing.

## Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

