import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Load the trained RandomForest model
model = joblib.load('ocular_disease_model.pkl')  # Adjust the path if needed

# Function to preprocess the image (resize and normalize it)
def preprocess_image(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Convert to RGB and resize (if needed)
    img = img.convert('RGB')
    img = img.resize((224, 224))  # Resize to match the input size of your model (you can adjust this)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0  # Normalize the pixel values
    return img_array.flatten()  # Flatten the image

# If you have patient information (age and sex) for input
def get_patient_data(age, sex):
    # Prepare the patient data (just the age and sex, for now)
    patient_data = np.array([[age, sex]])  # Example format, modify as needed

    # Example scaling (if you did scaling during training, use the same scaler)
    scaler = StandardScaler()
    patient_data_scaled = scaler.fit_transform(patient_data)
    return patient_data_scaled

# Prediction function
def predict_ocular_disease(patient_data, image_path):
    # Preprocess the image
    img_data = preprocess_image(image_path)
    
    # Ensure the image data has the correct number of features
    expected_image_features = 6  # Adjust this based on your model's requirements
    img_data = img_data[:expected_image_features]  # Trim the image data to match expected features

    # Reshape image data to ensure it's a 2D array (1 sample with many features)
    img_data = img_data.reshape(1, -1)  # Reshape into (1, number_of_features)
    
    # Concatenate the patient data and image data
    input_data = np.hstack((patient_data, img_data))  # Combine the two datasets
    
    # Ensure input_data matches the shape expected by the model
    if input_data.shape[1] != 8:  # Check if there are 8 features
        raise ValueError("Input data must have exactly 8 features (age, sex, and image data).")
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Decode the prediction (assuming model outputs labels like N, D, G, etc.)
    if prediction == 'N':
        result = "Normal"
    elif prediction == 'D':
        result = "Diabetes"
    elif prediction == 'G':
        result = "Eye Pressure"
    elif prediction == 'C':
        result = "Cataract"
    elif prediction == 'A':
        result = "Age-related Macular Degeneration"
    elif prediction == 'H':
        result = "Hypertension"
    elif prediction == 'M':
        result = "Pathological Myopia"
    else:
        result = "You have a risk of Glaucoma"
    
    return result

# Example usage
age = 45
sex = 1  # 1 for male, 0 for female (just an example)
image_path = r"C:\Users\shavy\OneDrive\Desktop\clinical documents\images (2).jpeg" # Corrected image path

# Get patient data (scale it if needed)
patient_data = get_patient_data(age, sex)

# Predict the ocular disease
result = predict_ocular_disease(patient_data, image_path)
print(f"Predicted Ocular Disease: {result}")
