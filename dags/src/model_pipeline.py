import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os



# Function to Pull and Load Preprocessed Data from DVC
def load_preprocessed_data(filename):
    !dvc pull {filename}
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        print(f"Loaded preprocessed data from {filename}.")
        return df
    else:
        print(f"File {filename} not found in DVC.")
        return None


def train_model(df, target_column="value"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize XGBoost Regressor
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42,reg_alpha=0.1, reg_lambda=1.0,learning_rate=0.05, n_estimators=1000, min_child_weight=5)

    # Train the model
    xgb_reg.fit(X_train, y_train)
    
    return xgb_reg, X_train, X_test, y_train, y_test

    

# Function to Test the Model
def test_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse}")
    print(f"Test R2: {r2}")
    return mse, r2

    

# Function to Validate the Model
def validate_model(model, X_train, y_train):
    # Perform validation on training data
    y_pred_train = model.predict(X_train)

    # Calculate performance metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    print(f"Validation MSE: {mse_train}")
    print(f"Validation R2: {r2_train}")
    return mse_train, r2_train


# Function to Upload Model to Model Store (Save as Pickle and Push to DVC)
def upload_to_model_store(model, model_name="xgb_reg.pkl"):
    # Save the model as a pickle file
    with open(model_name, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as {model_name}.")

    # Track the model file with DVC
    !dvc add {model_name}
    !git add {model_name}.dvc
    !git commit -m "Add trained model with DVC tracking"
    !git push origin main
    !dvc push
    print("Model uploaded to model store.")
    
    

# Function to Download Model from Model Store
def download_from_model_store(model_name="xgb_reg.pkl"):
    # Pull the model file from DVC
    !dvc pull {model_name}
    
    # Load the model if it exists
    if os.path.exists(model_name):
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
        print(f"Model {model_name} loaded from model store.")
        return model
    else:
        print("Model not found in model store.")
        return None


    
# Run the Complete Model Pipeline
def run_model_pipeline(target_column="value"):
    
    # Step 1: Run data processing and get the final preprocessed filename
    from data_processing import run_data_pipeline  # Import dynamically from data_processing.py
    final_data_filename = run_data_pipeline(df_raw)  # Pass raw data to start the pipeline

    # Step 2: Load the preprocessed data
    df = load_preprocessed_data(final_data_filename)

    if df is not None:
        # Step 3: Train the model
        model, X_train, X_test, y_train, y_test = train_model(df, target_column)
        
        # Step 4: Test the model
        test_model(model, X_test, y_test)
        
        # Step 5: Validate the model (optional)
        validate_model(model, X_train, y_train)
        
        # Step 6: Upload the trained model to the model store
        upload_to_model_store(model)
    else:
        print("Model pipeline aborted: Preprocessed data could not be loaded.")
