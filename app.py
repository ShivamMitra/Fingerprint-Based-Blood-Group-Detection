import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from typing import Dict, Tuple, List, Any
import os

# Constants
CLASS_NAMES = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
# Update these paths to match your actual file locations
MODEL_PATHS = {
    'AlexNet': r'Fingerprint Based Blood Group Detection\code\Alexnet\model_blood_group_detection_alextnet.keras',
    'LeNet': r'Fingerprint Based Blood Group Detection\code\Lenet\model_blood_group_detection_lenet.keras',
    'ResNet-34': r'Fingerprint Based Blood Group Detection\code\Resnet34\model_blood_group_detection.h5',
    'VGG-16': r'Fingerprint Based Blood Group Detection\code\Vgg16\blood_group_detection_vgg16.h5'
}

def get_model_config(model_name: str) -> dict:
    """Get configuration for each model type."""
    configs = {
        'alexnet': {
            'input_shape': (256, 256, 3),
            'preprocess_input': lambda x: x / 255.0
        },
        'lenet': {
            'input_shape': (224, 224, 3),  # LeNet expects grayscale
            'preprocess_input': lambda x: x / 255.0
        },
        'resnet': {
            'input_shape': (256, 256, 3),  # Updated to match model's expectation
            'preprocess_input': tf.keras.applications.resnet50.preprocess_input
        },
        'vgg': {
            'input_shape': (256, 256, 3),  # Updated to match model's expectation
            'preprocess_input': tf.keras.applications.vgg16.preprocess_input
        }
    }
    
    model_type = model_name.lower()
    if 'alexnet' in model_type:
        return configs['alexnet']
    elif 'lenet' in model_type:
        return configs['lenet']
    elif 'resnet' in model_type:
        return configs['resnet']
    else:  # vgg
        return configs['vgg']

def load_model(model_path: str, model_name: str) -> tf.keras.Model:
    """Load a model with proper input shape handling and optimizer compatibility."""
    try:
        # Get model config
        config = get_model_config(model_name)
        
        # Custom objects for loading models
        custom_objects = {
            'Adam': tf.keras.optimizers.Adam,  # Use standard Adam
            'AdamW': tf.keras.optimizers.Adam  # Fallback to Adam if AdamW is used
        }
        
        # Try to load the model
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
        except Exception as load_error:
            st.sidebar.warning(f"⚠️ Error loading {model_name}: {str(load_error)}")
            st.sidebar.info(f"Creating new {model_name} model...")
            model = create_model(model_name, len(CLASS_NAMES))
        
        # Rebuild model with correct input shape
        input_layer = tf.keras.layers.Input(shape=config['input_shape'])
        
        # Special handling for LeNet
        if 'lenet' in model_name.lower():
            # Convert input to grayscale if needed
            if config['input_shape'][-1] == 1:
                x = tf.keras.layers.Lambda(
                    lambda img: tf.image.rgb_to_grayscale(img),
                    name='rgb_to_grayscale'
                )(input_layer)
            else:
                x = input_layer
        else:
            x = input_layer
        
        # Get the output from the model
        output = model(x)
        
        # Create new model
        new_model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        # Compile the model with standard Adam
        new_model.compile(
            optimizer='adam',  # Use string identifier instead of optimizer instance
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return new_model
        
    except Exception as e:
        st.sidebar.error(f"Error loading {model_name}: {str(e)}")
        return None
    
def create_model(model_name: str, num_classes: int) -> tf.keras.Model:
    """Create a model from scratch if loading fails."""
    config = get_model_config(model_name)
    
    if 'lenet' in model_name.lower():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=config['input_shape']),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    else:
        # Fallback to a simple CNN
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=config['input_shape']),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

@st.cache_resource
def load_all_models() -> Dict[str, Tuple[tf.keras.Model, dict]]:
    """Load all models and return a dictionary of (model, config) tuples."""
    models = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            st.sidebar.error(f"❌ {name} model not found at: {path}")
            continue
            
        model = load_model(path, name)
        if model is not None:
            models[name] = (model, get_model_config(name))
            st.sidebar.success(f"✅ {name} loaded successfully")
    
    return models

def preprocess_image(image: np.ndarray, config: dict) -> np.ndarray:
    """Preprocess image according to model requirements."""
    # Convert to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[..., :3]
    
    # Resize
    target_size = config['input_shape'][:2]
    image = tf.image.resize(image, target_size).numpy()
    
    # Convert to grayscale if needed
    if config['input_shape'][-1] == 1 and image.shape[-1] == 3:
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        image = np.expand_dims(image, axis=-1)
    
    # Apply model-specific preprocessing
    image = config['preprocess_input'](image)
    
    return image

def plot_confidence(predictions: Dict[str, Any]) -> None:
    """Plot confidence levels for all blood groups."""
    # Get predictions from all models
    model_names = list(predictions.keys())
    if not model_names:
        st.warning("No predictions to display")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), 
                                 gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot confidence for each model
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(model_names)
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        if 'probabilities' in pred:
            probs = pred['probabilities']
            offset = width * i - (width * (len(model_names) - 1)) / 2
            bars = ax1.bar(x + offset, [p * 100 for p in probs], 
                          width, label=model_name)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 5:  # Only show labels for significant probabilities
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Customize plot
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('Confidence (%)')
    ax1.set_title('Blood Group Prediction Confidence')
    ax1.legend()
    
    # Add model agreement table
    table_data = []
    for model_name, pred in predictions.items():
        table_data.append([
            model_name,
            pred.get('prediction', 'N/A'),
            f"{pred.get('confidence', 0) * 100:.1f}%"
        ])
    
    # Create table
    table = ax2.table(
        cellText=table_data,
        colLabels=['Model', 'Prediction', 'Confidence'],
        loc='center',
        cellLoc='center'
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax2.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

def main():
    # Page setup
    st.set_page_config(
        page_title="Blood Group Classifier",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧪 Blood Group Classification")
    st.markdown("Upload a fingerprint image to predict the blood group using our ensemble model.")
    
    # Load models
    st.sidebar.header("Model Status")
    models = load_all_models()
    
    if not models:
        st.error("""
        ⚠️ No models could be loaded. Please ensure:
        1. Model files exist in the correct locations
        2. File paths in MODEL_PATHS are correct
        3. You have the required dependencies installed
        """)
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a fingerprint image",
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded_file is not None:
        try:
            # Load and preprocess image
            image = Image.open(uploaded_file)
            image = np.array(image.convert('RGB'))
            
            # Display original image
            st.subheader("Uploaded Fingerprint")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, use_column_width=True)
            
            with col2:
                # Get predictions from all models
                predictions = {}
                
                for model_name, (model, config) in models.items():
                    try:
                        # Preprocess image for the model
                        processed_img = preprocess_image(image, config)
                        processed_img = np.expand_dims(processed_img, axis=0)
                        
                        # Make prediction
                        pred = model.predict(processed_img, verbose=0)
                        pred = pred.flatten()  # Ensure it's 1D
                        
                        # Store prediction results
                        pred_class = np.argmax(pred)
                        confidence = float(pred[pred_class])
                        
                        predictions[model_name] = {
                            'prediction': CLASS_NAMES[pred_class],
                            'confidence': confidence,
                            'probabilities': pred.tolist()
                        }
                        
                    except Exception as e:
                        st.sidebar.error(f"Error in {model_name} prediction: {str(e)}")
                        predictions[model_name] = {
                            'prediction': 'Error',
                            'confidence': 0.0,
                            'probabilities': [0.0] * len(CLASS_NAMES),
                            'error': str(e)
                        }
                
                # Calculate ensemble prediction
                if predictions:
                    # Get all successful predictions
                    valid_preds = {k: v for k, v in predictions.items() 
                                 if 'error' not in v}
                    
                    if valid_preds:
                        # Calculate average probabilities
                        avg_probs = np.mean(
                            [p['probabilities'] for p in valid_preds.values()],
                            axis=0
                        )
                        ensemble_pred = CLASS_NAMES[np.argmax(avg_probs)]
                        ensemble_conf = float(np.max(avg_probs))
                        
                        # Count agreement
                        agreement = sum(1 for p in valid_preds.values() 
                                      if p['prediction'] == ensemble_pred)
                        
                        # Display ensemble result
                        st.subheader("Ensemble Prediction")
                        st.metric(
                            "Predicted Blood Group",
                            ensemble_pred,
                            f"{ensemble_conf*100:.1f}% confidence"
                        )
                        st.caption(f"Agreement: {agreement} out of {len(valid_preds)} models agree")
            
            # Plot confidence levels
            if predictions:
                st.subheader("Model Predictions")
                plot_confidence(predictions)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)  # Show full error for debugging

if __name__ == "__main__":
    main()