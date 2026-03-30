import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from typing import Dict, Tuple, List, Any
import os
import cv2
import json
from datetime import datetime
import webbrowser
import tempfile
import shutil
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

# Constants
CLASS_NAMES = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
# Update these paths to match your actual file locations
MODEL_PATHS = {
    'AlexNet': r'Fingerprint Based Blood Group Detection\code\Alexnet\model_blood_group_detection_alextnet.keras',
    'LeNet': r'Fingerprint Based Blood Group Detection\code\Lenet\model_blood_group_detection_lenet.keras',
    'ResNet-34': r'Fingerprint Based Blood Group Detection\code\Resnet34\model_blood_group_detection.h5',
    'VGG-16': r'Fingerprint Based Blood Group Detection\code\Vgg16\blood_group_detection_vgg16.h5'
}

# XAI Model configurations (from xai.py)
XAI_MODEL_CONFIGS = {
    'ResNet-34': {
        'path': os.path.join('Fingerprint Based Blood Group Detection', 'code', 'Resnet34', 'model_blood_group_detection.h5'),
        'layer': 'conv5_block3_out',
        'input_size': (256, 256),
        'description': 'Deep residual network with 34 layers'
    },
    'VGG-16': {
        'path': os.path.join('Fingerprint Based Blood Group Detection', 'code', 'Vgg16', 'blood_group_detection_vgg16.h5'),
        'layer': 'block5_conv3',
        'input_size': (256, 256),
        'description': 'Very deep convolutional network with 16 layers'
    },
    'LeNet': {
        'path': os.path.join('Fingerprint Based Blood Group Detection', 'code', 'Lenet', 'model_blood_group_detection_lenet.keras'),
        'layer': None,
        'input_size': (224, 224),
        'description': 'Classic convolutional neural network'
    },
    'AlexNet': {
        'path': os.path.join('Fingerprint Based Blood Group Detection', 'code', 'Alexnet', 'model_blood_group_detection_alextnet.keras'),
        'layer': None,
        'input_size': (227, 227),
        'description': 'Deep convolutional network with 8 layers'
    }
}

# Enhanced color palette for visualizations
COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ff7f0e',
    'danger': '#d62728',
    'light': '#f8f9fa',
    'dark': '#2c3e50',
    'purple': '#9467bd',
    'orange': '#ff7f0e',
    'cyan': '#17becf'
}

class XAIAnalyzer:
    """XAI functionality from xai.py integrated into Streamlit"""
    
    def __init__(self):
        self.models = {}
        self.prediction_results = {}
        self.xai_results = {}
        self.image_path = None
        
    def clean_path(self, path):
        if not path:
            return path
        path = str(path).strip().strip('"\'')
        while '//' in path:
            path = path.replace('//', '/')
        while '\\\\' in path:
            path = path.replace('\\\\', '\\')
        try:
            return os.path.normpath(os.path.abspath(path))
        except:
            return path
    
    def convert_to_functional(self, model, input_shape, model_name):
        """Convert Sequential model to Functional model with unique layer names."""
        try:
            inputs = tf.keras.Input(shape=input_shape, name=f'{model_name}_input')
            
            x = inputs
            for i, layer in enumerate(model.layers):
                config = layer.get_config()
                config['name'] = f'{model_name}_layer_{i}_{config.get("name", "layer")}'
                new_layer = type(layer).from_config(config)
                x = new_layer(x)
            
            functional_model = tf.keras.Model(inputs=inputs, outputs=x, name=f'{model_name}_functional')
            
            sample = np.zeros((1,) + input_shape, dtype=np.float32)
            _ = functional_model(sample)
            
            return functional_model
        except Exception as e:
            print(f"Error converting to functional: {str(e)}")
            return model
    
    def load_xai_models(self):
        """Load models for XAI analysis"""
        for model_name, config in XAI_MODEL_CONFIGS.items():
            try:
                model_path = self.clean_path(config['path'])
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path, compile=False)
                    
                    if isinstance(model, tf.keras.Sequential):
                        model = self.convert_to_functional(model, config['input_size'] + (3,), model_name)
                    
                    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    self.models[model_name] = {'model': model, 'config': config}
            except Exception as e:
                print(f"Error loading XAI model {model_name}: {str(e)}")
    
    def find_last_conv_layer(self, model):
        """Find the last convolutional layer in the model."""
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        return None
    
    def generate_gradcam_accurate(self, model, image, class_index):
        """Generate accurate Grad-CAM heatmap with improved precision."""
        try:
            # Find the last convolutional layer
            last_conv_layer_name = self.find_last_conv_layer(model)
            if last_conv_layer_name is None:
                print("No convolutional layer found in model")
                return None
            
            # Create a model that outputs both conv features and predictions
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(last_conv_layer_name).output, model.output]
            )
            
            # Convert image to tensor if needed
            if not tf.is_tensor(image):
                image = tf.convert_to_tensor(image, dtype=tf.float32)
            
            # Compute gradients using GradientTape
            with tf.GradientTape() as tape:
                # Watch the input image
                tape.watch(image)
                
                # Get the conv features and predictions
                conv_outputs, predictions = grad_model(image)
                
                # Get the loss for the target class
                loss = predictions[:, class_index]
            
            # Get gradients of the loss with respect to conv outputs
            grads = tape.gradient(loss, conv_outputs)
            
            if grads is None:
                print("Gradients are None")
                return None
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the conv features with the pooled gradients
            conv_outputs = conv_outputs[0]  # Remove batch dimension
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Apply ReLU and normalize with better precision
            heatmap = tf.maximum(heatmap, 0)
            heatmap_max = tf.reduce_max(heatmap)
            if heatmap_max > 0:
                heatmap = heatmap / heatmap_max
            
            # Convert to numpy
            heatmap = heatmap.numpy()
            
            # Resize to match original image size with better interpolation
            heatmap = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_CUBIC)
            
            # Apply Gaussian blur for smoother visualization
            heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
            
            # Convert to uint8 and apply better colormap
            heatmap = np.uint8(255 * heatmap)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Create overlay with original image for better context
            original_img = image[0].numpy() * 255
            original_img = original_img.astype(np.uint8)
            original_img = cv2.resize(original_img, (256, 256))
            
            # Blend with alpha for better visibility - FIXED: Added gamma parameter
            alpha = 0.4
            overlay = cv2.addWeighted(original_img, 1-alpha, heatmap_colored, alpha, 0)  # Added gamma=0
            
            return overlay
            
        except Exception as e:
            print(f"Grad-CAM generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_occlusion_accurate(self, model, image, class_index, patch_size=8):
        """Generate accurate occlusion sensitivity map with better precision."""
        try:
            # Convert image to numpy if it's a tensor
            if tf.is_tensor(image):
                image_np = image.numpy()
            else:
                image_np = image
            
            # Handle different image formats
            if len(image_np.shape) == 4:
                image_np = image_np[0]  # Remove batch dimension
            elif len(image_np.shape) == 3:
                pass  # Already in correct format (H, W, C)
            else:
                print(f"Unexpected image shape: {image_np.shape}")
                return None
            
            h, w = image_np.shape[:2]
            
            # Get original prediction
            if len(image_np.shape) == 3:
                pred_input = np.expand_dims(image_np, axis=0)
            else:
                pred_input = image_np
            
            original_pred = model.predict(pred_input, verbose=0)
            original_confidence = original_pred[0, class_index]
            
            # Initialize occlusion map with higher resolution
            occlusion_map = np.zeros((h, w))
            
            # Create a copy of the original image
            original_image = image_np.copy()
            
            # Iterate over patches with smaller size for better precision
            step_size = patch_size
            for i in range(0, h, step_size):
                for j in range(0, w, step_size):
                    # Create occluded image
                    occluded_image = original_image.copy()
                    
                    # Define patch boundaries
                    i_end = min(i + patch_size, h)
                    j_end = min(j + patch_size, w)
                    
                    # Apply occlusion with Gaussian noise instead of mean
                    patch_mean = np.mean(original_image[i:i_end, j:j_end])
                    patch_std = np.std(original_image[i:i_end, j:j_end])
                    noise = np.random.normal(patch_mean, patch_std * 0.5, (i_end-i, j_end-j, 3))
                    occluded_image[i:i_end, j:j_end] = np.clip(noise, 0, 255)
                    
                    # Prepare for prediction
                    occluded_input = np.expand_dims(occluded_image, axis=0)
                    
                    # Get prediction with occlusion
                    try:
                        occluded_pred = model.predict(occluded_input, verbose=0)
                        occluded_confidence = occluded_pred[0, class_index]
                        
                        # Calculate sensitivity (difference in confidence)
                        sensitivity = original_confidence - occluded_confidence
                        
                        # Fill the occlusion map patch with sensitivity value
                        occlusion_map[i:i_end, j:j_end] = sensitivity
                    except Exception as pred_error:
                        print(f"Prediction error at patch ({i},{j}): {pred_error}")
                        continue
            
            # Apply Gaussian smoothing for better visualization
            occlusion_map = cv2.GaussianBlur(occlusion_map, (3, 3), 0)
            
            # Normalize the occlusion map with better handling
            occlusion_map = occlusion_map - occlusion_map.min()
            occlusion_map_max = occlusion_map.max()
            if occlusion_map_max > 0:
                occlusion_map = occlusion_map / occlusion_map_max
            
            # Convert to uint8 for visualization
            occlusion_map = np.uint8(255 * occlusion_map)
            
            # Apply better colormap for visualization
            occlusion_colored = cv2.applyColorMap(occlusion_map, cv2.COLORMAP_HOT)
            occlusion_colored = cv2.cvtColor(occlusion_colored, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size with better interpolation
            occlusion_colored = cv2.resize(occlusion_colored, (256, 256), interpolation=cv2.INTER_CUBIC)
            
            return occlusion_colored
            
        except Exception as e:
            print(f"Occlusion generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_smoothgrad_accurate(self, model, image, class_index, n_samples=10, noise_level=0.1):
        """Generate accurate SmoothGrad visualization with better precision."""
        try:
            # Convert image to tensor if needed
            if not tf.is_tensor(image):
                image = tf.convert_to_tensor(image, dtype=tf.float32)
            
            gradients = []
            
            for _ in range(n_samples):
                # Add Gaussian noise
                noise = tf.random.normal(shape=tf.shape(image), stddev=noise_level, dtype=tf.float32)
                noisy_image = image + noise
                noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
                
                # Compute gradients
                with tf.GradientTape() as tape:
                    tape.watch(noisy_image)
                    predictions = model(noisy_image)
                    loss = predictions[:, class_index]
                
                grad = tape.gradient(loss, noisy_image)
                if grad is not None:
                    gradients.append(grad)
            
            if not gradients:
                print("No gradients computed")
                return None
            
            # Average the gradients
            avg_gradient = tf.reduce_mean(gradients, axis=0)
            
            # Take absolute value and reduce across channels
            avg_gradient = tf.reduce_mean(tf.abs(avg_gradient), axis=-1)
            avg_gradient = avg_gradient[0]  # Remove batch dimension
            
            # Normalize the gradient with better precision
            grad_min = tf.reduce_min(avg_gradient)
            grad_max = tf.reduce_max(avg_gradient)
            grad_range = grad_max - grad_min
            
            if grad_range > 0:
                avg_gradient = (avg_gradient - grad_min) / grad_range
            
            # Convert to numpy and resize with better interpolation
            avg_gradient = avg_gradient.numpy()
            avg_gradient = cv2.resize(avg_gradient, (256, 256), interpolation=cv2.INTER_CUBIC)
            
            # Apply Gaussian smoothing for better visualization
            avg_gradient = cv2.GaussianBlur(avg_gradient, (3, 3), 0)
            
            # Convert to uint8 and apply better colormap
            avg_gradient = np.uint8(255 * avg_gradient)
            gradient_colored = cv2.applyColorMap(avg_gradient, cv2.COLORMAP_VIRIDIS)
            gradient_colored = cv2.cvtColor(gradient_colored, cv2.COLOR_BGR2RGB)
            
            return gradient_colored
            
        except Exception as e:
            print(f"SmoothGrad generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_xai_visualizations_accurate(self, model_name, image, class_index):
        """Generate all accurate XAI visualizations for a model."""
        model_info = self.models.get(model_name)
        if not model_info:
            return None
        
        model = model_info['model']
        visualizations = {}
        
        # Accurate Grad-CAM
        gradcam = self.generate_gradcam_accurate(model, image, class_index)
        if gradcam is not None:
            visualizations['gradcam'] = gradcam
        
        # Accurate SmoothGrad
        smoothgrad = self.generate_smoothgrad_accurate(model, image, class_index)
        if smoothgrad is not None:
            visualizations['smoothgrad'] = smoothgrad
        
        # Accurate Occlusion
        occlusion = self.generate_occlusion_accurate(model, image, class_index)
        if occlusion is not None:
            visualizations['occlusion'] = occlusion
        
        return visualizations
    
    def create_xai_grid_accurate(self, original_image, model_name, xai_viz):
        """Create an accurate grid visualization with original image and XAI results."""
        if not xai_viz:
            return None
        
        # Create figure with better layout
        fig = plt.figure(figsize=(16, 12), facecolor='white')
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
        
        fig.suptitle(f'🔍 XAI Analysis - {model_name}', fontsize=18, fontweight='bold', color=COLOR_PALETTE['dark'])
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_image)
        ax1.set_title('📸 Original Fingerprint', fontsize=14, fontweight='bold', color=COLOR_PALETTE['primary'])
        ax1.axis('off')
        
        # Grad-CAM
        ax2 = fig.add_subplot(gs[0, 1])
        if 'gradcam' in xai_viz:
            ax2.imshow(xai_viz['gradcam'])
            ax2.set_title('🎯 Grad-CAM\n(Important Regions)', fontsize=12, color=COLOR_PALETTE['success'])
        else:
            ax2.text(0.5, 0.5, 'Grad-CAM\nNot Available', 
                           ha='center', va='center', transform=ax2.transAxes,
                           fontsize=12, color=COLOR_PALETTE['warning'])
        ax2.axis('off')
        
        # Occlusion Sensitivity
        ax3 = fig.add_subplot(gs[1, 0])
        if 'occlusion' in xai_viz:
            ax3.imshow(xai_viz['occlusion'])
            ax3.set_title('🔒 Occlusion Sensitivity\n(Critical Areas)', fontsize=12, color=COLOR_PALETTE['secondary'])
        else:
            ax3.text(0.5, 0.5, 'Occlusion Sensitivity\nNot Available', 
                           ha='center', va='center', transform=ax3.transAxes,
                           fontsize=12, color=COLOR_PALETTE['warning'])
        ax3.axis('off')
        
        # SmoothGrad
        ax4 = fig.add_subplot(gs[1, 1])
        if 'smoothgrad' in xai_viz:
            ax4.imshow(xai_viz['smoothgrad'])
            ax4.set_title('🌊 SmoothGrad\n(Gradient Analysis)', fontsize=12, color=COLOR_PALETTE['primary'])
        else:
            ax4.text(0.5, 0.5, 'SmoothGrad\nNot Available', 
                           ha='center', va='center', transform=ax4.transAxes,
                           fontsize=12, color=COLOR_PALETTE['warning'])
        ax4.axis('off')
        
        # Add styling to all axes
        for ax in [ax1, ax2, ax3, ax4]:
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_PALETTE['light'])
                spine.set_linewidth(2)
        
        plt.tight_layout()
        return fig

def get_model_config(model_name: str) -> dict:
    """Get configuration for each model type."""
    configs = {
        'alexnet': {
            'input_shape': (256, 256, 3),
            'preprocess_input': lambda x: x / 255.0
        },
        'lenet': {
            'input_shape': (224, 224, 3),
            'preprocess_input': lambda x: x / 255.0
        },
        'resnet': {
            'input_shape': (256, 256, 3),
            'preprocess_input': tf.keras.applications.resnet50.preprocess_input
        },
        'vgg': {
            'input_shape': (256, 256, 3),
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
    else:
        return configs['vgg']

def load_model(model_path: str, model_name: str) -> tf.keras.Model:
    """Load a model with proper input shape handling and optimizer compatibility."""
    try:
        config = get_model_config(model_name)
        
        custom_objects = {
            'Adam': tf.keras.optimizers.Adam,
            'AdamW': tf.keras.optimizers.Adam
        }
        
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
        
        input_layer = tf.keras.layers.Input(shape=config['input_shape'])
        
        if 'lenet' in model_name.lower():
            if config['input_shape'][-1] == 1:
                x = tf.keras.layers.Lambda(
                    lambda img: tf.image.rgb_to_grayscale(img),
                    name='rgb_to_grayscale'
                )(input_layer)
            else:
                x = input_layer
        else:
            x = input_layer
        
        output = model(x)
        new_model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        new_model.compile(
            optimizer='adam',
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
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[..., :3]
    
    target_size = config['input_shape'][:2]
    image = tf.image.resize(image, target_size).numpy()
    
    if config['input_shape'][-1] == 1 and image.shape[-1] == 3:
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        image = np.expand_dims(image, axis=-1)
    
    image = config['preprocess_input'](image)
    
    return image

def plot_confidence_accurate(predictions: Dict[str, Any]) -> None:
    """Create an accurate confidence plot using Plotly with better visualization."""
    model_names = list(predictions.keys())
    if not model_names:
        st.warning("No predictions to display")
        return
    
    # Prepare data for better visualization
    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['success'], COLOR_PALETTE['purple']]
    
    # Create accurate bar chart
    fig_bar = go.Figure()
    
    for i, model_name in enumerate(predictions.keys()):
        if 'probabilities' in predictions[model_name]:
            probs = predictions[model_name]['probabilities']
            color = colors[i % len(colors)]
            
            fig_bar.add_trace(
                go.Bar(
                    name=model_name,
                    x=CLASS_NAMES,
                    y=[p * 100 for p in probs],
                    text=[f'{p*100:.1f}%' for p in probs],
                    textposition='outside',
                    marker=dict(
                        color=color,
                        line=dict(width=2, color='white'),
                        pattern_shape=None
                    ),
                    opacity=0.8
                )
            )
    
    fig_bar.update_layout(
        title=dict(
            text="Blood Group Prediction Confidence Analysis",
            x=0.5,
            font=dict(size=20, color=COLOR_PALETTE['dark'])
        ),
        xaxis=dict(
            title_text="Blood Group",
            title_font=dict(size=14, color=COLOR_PALETTE['dark']),
            tickangle=-45,
            title_standoff=25,
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title_text="Confidence (%)",
            title_font=dict(size=14, color=COLOR_PALETTE['dark']),
            range=[0, 110],
            gridcolor='lightgray',
            showgrid=True
        ),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Create accurate radar chart for model comparison
    fig_radar = go.Figure()
    
    for i, model_name in enumerate(predictions.keys()):
        if 'probabilities' in predictions[model_name]:
            probs = predictions[model_name]['probabilities']
            color = colors[i % len(colors)]
            
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[p * 100 for p in probs],
                    theta=CLASS_NAMES,
                    fill='toself',
                    name=model_name,
                    line=dict(color=color, width=2),
                    opacity=0.7
                )
            )
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%',
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                categoryorder='array',
                categoryarray=CLASS_NAMES
            )
        ),
        title=dict(
            text="Model Comparison Radar Chart",
            x=0.5,
            font=dict(size=16, color=COLOR_PALETTE['dark'])
        ),
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor='white'
    )
    
    # Create detailed table
    fig_table = go.Figure()
    
    table_data = []
    for model_name, pred in predictions.items():
        confidence = pred.get('confidence', 0) * 100
        prediction = pred.get('prediction', 'N/A')
        
        # Add confidence level indicator
        if confidence > 70:
            confidence_indicator = "🟢 High"
        elif confidence > 40:
            confidence_indicator = "🟡 Medium"
        else:
            confidence_indicator = "🔴 Low"
        
        table_data.append([
            f"🤖 {model_name}",
            prediction,
            f"{confidence:.1f}%",
            confidence_indicator
        ])
    
    fig_table.add_trace(
        go.Table(
            header=dict(
                values=['<b>Model</b>', '<b>Prediction</b>', '<b>Confidence</b>', '<b>Performance</b>'],
                fill_color=COLOR_PALETTE['light'],
                align='left',
                font=dict(size=12, color=COLOR_PALETTE['dark']),
                height=40
            ),
            cells=dict(
                values=table_data,
                fill_color=['white', 'rgba(240,240,240,0.5)', 'white', 'rgba(240,240,240,0.5)'],
                align='left',
                font=dict(size=11),
                height=30
            )
        )
    )
    
    fig_table.update_layout(
        title=dict(
            text="Detailed Prediction Results",
            x=0.5,
            font=dict(size=16, color=COLOR_PALETTE['dark'])
        ),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='white'
    )
    
    # Display all visualizations
    st.plotly_chart(fig_bar, use_container_width=True)
    st.plotly_chart(fig_radar, use_container_width=True)
    st.plotly_chart(fig_table, use_container_width=True)

def display_xai_results_accurate(original_image, predictions, models):
    """Display accurate XAI visualizations for all models."""
    st.subheader("🔍 Explainable AI (XAI) Analysis")
    
    # Initialize XAI analyzer
    xai_analyzer = XAIAnalyzer()
    xai_analyzer.load_xai_models()
    
    # Create tabs for each model
    model_names = [name for name in predictions.keys() if 'error' not in predictions[name]]
    
    if not model_names:
        st.warning("No successful predictions available for XAI analysis")
        return
    
    tabs = st.tabs(model_names)
    
    for i, model_name in enumerate(model_names):
        with tabs[i]:
            pred = predictions[model_name]
            class_index = CLASS_NAMES.index(pred['prediction'])
            
            # Get the correct model for XAI
            xai_model_name = model_name
            if 'ResNet' in model_name:
                xai_model_name = 'ResNet-34'
            elif 'VGG' in model_name:
                xai_model_name = 'VGG-16'
            elif 'AlexNet' in model_name:
                xai_model_name = 'AlexNet'
            elif 'LeNet' in model_name:
                xai_model_name = 'LeNet'
            
            # Get XAI model config
            xai_config = XAI_MODEL_CONFIGS.get(xai_model_name)
            if not xai_config:
                st.warning(f"XAI configuration not found for {model_name}")
                continue
            
            # Preprocess image for XAI
            xai_image = np.array(original_image)
            xai_image = cv2.resize(xai_image, xai_config['input_size'])
            xai_image = xai_image / 255.0
            xai_image = np.expand_dims(xai_image, axis=0)
            
            # Generate accurate XAI visualizations
            with st.spinner(f"🔄 Generating accurate XAI visualizations for {model_name}..."):
                xai_viz = xai_analyzer.generate_xai_visualizations_accurate(xai_model_name, xai_image, class_index)
            
            if xai_viz:
                # Display model prediction info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    confidence = pred['confidence'] * 100
                    confidence_color = COLOR_PALETTE['success'] if confidence > 70 else COLOR_PALETTE['warning'] if confidence > 40 else COLOR_PALETTE['danger']
                    
                    st.markdown(f"""
                        <div style='background-color: {COLOR_PALETTE['light']}; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 5px solid {confidence_color};'>
                            <h3 style='color: {COLOR_PALETTE['dark']}; margin: 0;'>🎯 {pred['prediction']}</h3>
                            <p style='color: {confidence_color}; font-size: 24px; font-weight: bold; margin: 10px 0;'>{confidence:.1f}%</p>
                            <p style='color: {COLOR_PALETTE['secondary']}; margin: 0;'>Confidence Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    confidence_level = "High" if pred['confidence'] > 0.8 else "Medium" if pred['confidence'] > 0.6 else "Low"
                    st.metric(
                        "Performance",
                        confidence_level,
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Class Index",
                        class_index,
                        delta=None
                    )
                
                # Create and display accurate XAI grid
                st.markdown("### 🎨 XAI Visualization Grid")
                fig = xai_analyzer.create_xai_grid_accurate(original_image, model_name, xai_viz)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Display individual visualizations with better layout
                st.markdown("### 🔬 Individual XAI Methods")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'gradcam' in xai_viz:
                        st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-left: 4px solid #2ca02c;'>
                            <h4 style='color: #2ca02c; margin: 0 0 10px 0;'>🎯 Grad-CAM</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(xai_viz['gradcam'], caption="Important regions identified by Grad-CAM", use_container_width=True)
                        st.caption("📌 Shows which regions of the fingerprint were most important for the prediction")
                
                with col2:
                    if 'occlusion' in xai_viz:
                        st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-left: 4px solid #ff7f0e;'>
                            <h4 style='color: #ff7f0e; margin: 0 0 10px 0;'>🔒 Occlusion Sensitivity</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(xai_viz['occlusion'], caption="Critical areas identified by Occlusion", use_container_width=True)
                        st.caption("🔍 Shows how prediction changes when different parts of the image are blocked")
                
                with col3:
                    if 'smoothgrad' in xai_viz:
                        st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-left: 4px solid #1f77b4;'>
                            <h4 style='color: #1f77b4; margin: 0 0 10px 0;'>🌊 SmoothGrad</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(xai_viz['smoothgrad'], caption="Gradient analysis by SmoothGrad", use_container_width=True)
                        st.caption("📈 Shows smooth gradient-based explanations of the model's decision")
            else:
                st.warning(f"❌ XAI visualizations not available for {model_name}")
                st.info("This might be due to model architecture limitations or preprocessing issues")

def main():
    # Enhanced page setup
    st.set_page_config(
        page_title="🧪 Blood Group Classifier with XAI",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown(f"""
    <style>
        .main-header {{
            font-size: 2.5rem;
            color: {COLOR_PALETTE['dark']};
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(135deg, {COLOR_PALETTE['light']} 0%, {COLOR_PALETTE['primary']} 100%);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card {{
            background-color: {COLOR_PALETTE['light']};
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 1rem 0;
            border-left: 5px solid {COLOR_PALETTE['primary']};
        }}
        .model-result {{
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 5px solid {COLOR_PALETTE['success']};
        }}
        .high-confidence {{ border-left-color: {COLOR_PALETTE['success']}; }}
        .medium-confidence {{ border-left-color: {COLOR_PALETTE['warning']}; }}
        .low-confidence {{ border-left-color: {COLOR_PALETTE['danger']}; }}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧪 Blood Group Classification with XAI</h1>', unsafe_allow_html=True)
    st.markdown("🔬 Upload a fingerprint image to predict blood group using our ensemble model with accurate XAI visualizations.")
    
    # Load models
    st.sidebar.header("🤖 Model Status")
    models = load_all_models()
    
    if not models:
        st.error(f"""
        ⚠️ No models could be loaded. Please ensure:
        1. Model files exist in the correct locations
        2. File paths in MODEL_PATHS are correct
        3. You have the required dependencies installed
        """)
        return
    
    # Enhanced file uploader
    uploaded_file = st.file_uploader(
        "📸 Choose a fingerprint image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear fingerprint image for blood group prediction"
    )
    
    if uploaded_file is not None:
        try:
            # Load and preprocess image
            image = Image.open(uploaded_file)
            image = np.array(image.convert('RGB'))
            
            # Display original image with enhanced styling
            st.subheader("📸 Uploaded Fingerprint")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("""
                    <div style='background-color: white; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
                        <h4 style='color: #2c3e50; margin: 0;'>Original Image</h4>
                    </div>
                    """, unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            
            with col2:
                # Get predictions from all models
                predictions = {}
                
                for model_name, (model, config) in models.items():
                    try:
                        processed_img = preprocess_image(image, config)
                        processed_img = np.expand_dims(processed_img, axis=0)
                        
                        pred = model.predict(processed_img, verbose=0)
                        pred = pred.flatten()
                        
                        pred_class = np.argmax(pred)
                        confidence = float(pred[pred_class])
                        
                        predictions[model_name] = {
                            'prediction': CLASS_NAMES[pred_class],
                            'confidence': confidence,
                            'probabilities': pred.tolist()
                        }
                        
                    except Exception as e:
                        st.sidebar.error(f"❌ Error in {model_name} prediction: {str(e)}")
                        predictions[model_name] = {
                            'prediction': 'Error',
                            'confidence': 0.0,
                            'probabilities': [0.0] * len(CLASS_NAMES),
                            'error': str(e)
                        }
                
                # Calculate ensemble prediction
                if predictions:
                    valid_preds = {k: v for k, v in predictions.items() 
                                 if 'error' not in v}
                    
                    if valid_preds:
                        avg_probs = np.mean(
                            [p['probabilities'] for p in valid_preds.values()],
                            axis=0
                        )
                        ensemble_pred = CLASS_NAMES[np.argmax(avg_probs)]
                        ensemble_conf = float(np.max(avg_probs))
                        
                        agreement = sum(1 for p in valid_preds.values() 
                                      if p['prediction'] == ensemble_pred)
                        
                        # Display ensemble result with enhanced styling
                        st.subheader("🏆 Ensemble Prediction")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <h2 style='color: {COLOR_PALETTE['dark']}; margin: 0;'>{ensemble_pred}</h2>
                                    <p style='color: {COLOR_PALETTE['secondary']}; font-size: 18px; margin: 10px 0 0;'>Predicted Blood Group</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            st.metric(
                                "📊 Confidence",
                                f"{ensemble_conf*100:.1f}%",
                                delta=None
                            )
                        
                        with col3:
                            st.metric(
                                "🤝 Agreement",
                                f"{agreement}/{len(valid_preds)}",
                                delta=None
                            )
            
            # Plot accurate confidence levels
            if predictions:
                st.subheader("📊 Model Predictions")
                plot_confidence_accurate(predictions)
                
                # Display accurate XAI results
                display_xai_results_accurate(image, predictions, models)
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()