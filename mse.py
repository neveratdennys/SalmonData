import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    # Load the dataset (assuming the dataset is stored in a CSV file)
    data = pd.read_csv('cs-tr-pac-dfo-mpo-science-eng.csv', skiprows = 1)  # Replace 'salmon_data.csv' with your dataset's filename
    # Data Preprocessing (handle missing values, encoding categorical variables if needed)
    # data.columns = data.columns.str.strip()
    data = data.loc[data['NOTES'] != 'Catch figures omitted due to privacy restrictions.']

    features = ['LICENCE_AREA','MGMT_AREA', 'CALENDAR_YEAR', 'VESSEL_COUNT', 'BOAT_DAYS']

    breeds = ['SOCKEYE', 'COHO', 'PINK', 'CHUM', 'STEELHEAD', 'CHINOOK']

    unpivoted_data_by_breed = pd.DataFrame()

    for breed in breeds:
        x = data[features]
        x['KEPT'] = data[breed + '_KEPT']
        x['RELD'] = data[breed + '_RELD']
        x['BREED'] = breed

        unpivoted_data_by_breed = pd.concat([unpivoted_data_by_breed, x], ignore_index=True)

    unpivoted_data_by_breed = unpivoted_data_by_breed.loc[unpivoted_data_by_breed['BREED'] == 'PINK']
    
    unpivoted_data_by_breed['KEPT_CUMSUM'] = unpivoted_data_by_breed\
                                            .sort_values(by=['CALENDAR_YEAR'])\
                                            .groupby(['LICENCE_AREA', 'VESSEL_COUNT', 'BOAT_DAYS'])['KEPT']\
                                            .cumsum()
    unpivoted_data_by_breed['RELD_CUMSUM'] = unpivoted_data_by_breed.sort_values(by=['CALENDAR_YEAR']).groupby(['LICENCE_AREA', 'VESSEL_COUNT', 'BOAT_DAYS'])['RELD'].cumsum()
 
    train_df, test_df = train_test_split(unpivoted_data_by_breed, test_size=0.2, random_state=42)

    # Define independent variables (features) and dependent variables
    # features = ['LICENCE_AREA','MGMT_AREA', 'CALENDAR_YEAR', 'VESSEL_COUNT', 'BOAT_DAYS']
    # target_variables = ['SOCKEYE_KEPT', 'SOCKEYE_RELD']



    categorical_features = ['LICENCE_AREA', 'MGMT_AREA', 'BREED']
    target_variables = ['KEPT', 'RELD']

    scaled_features = ['CALENDAR_YEAR']

    scaled_features_transformer = Pipeline(
        steps=[
            ("calendar_year", MinMaxScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers = [
            ('CATEGORICAL_FEATURES', categorical_transformer, categorical_features),
        ],
        remainder='passthrough'
    )

    features = ['LICENCE_AREA', 'MGMT_AREA', 'CALENDAR_YEAR', 'VESSEL_COUNT', 'BOAT_DAYS', 'BREED', 'KEPT_CUMSUM', 'RELD_CUMSUM']

    
    
    # Split the data into features and target variables
    # X = data[features]
    # y = data[target_variables]

    # y['SOCKEYE_TOTAL'] = y['SOCKEYE_KEPT'] + y['SOCKEYE_RELD']

    # Partition the data into training and testing sets (e.g., 80% train, 20% test)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = preprocessor.fit_transform(train_df[features])

    y_train = train_df[target_variables]

    X_test = preprocessor.transform(test_df[features])
    y_test = test_df[target_variables]
    # Initialize and train the regression model

    model = LinearRegression()

    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model's accuracy (for example, using Mean Squared Error)
    mse = mean_squared_error(y_test, predictions)
    rsq = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"R squared: {rsq}")

if __name__ == "__main__":
    main()