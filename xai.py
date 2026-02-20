import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import json
import webbrowser

# Define model paths and configurations
MODEL_CONFIGS = {
    'ResNet34': {
        'path': os.path.join('Fingerprint Based Blood Group Detection', 'code', 'Resnet34', 'model_blood_group_detection.h5'),
        'input_size': (256, 256),
        'description': 'Deep residual network with 34 layers'
    },
    'VGG16': {
        'path': os.path.join('Fingerprint Based Blood Group Detection', 'code', 'Vgg16', 'blood_group_detection_vgg16.h5'),
        'input_size': (256, 256),
        'description': 'Very deep convolutional network with 16 layers'
    },
    'LeNet': {
        'path': os.path.join('Fingerprint Based Blood Group Detection', 'code', 'Lenet', 'model_blood_group_detection_lenet.keras'),
        'input_size': (224, 224),
        'description': 'Classic convolutional neural network'
    },
    'AlexNet': {
        'path': os.path.join('Fingerprint Based Blood Group Detection', 'code', 'Alexnet', 'model_blood_group_detection_alextnet.keras'),
        'input_size': (227, 227),
        'description': 'Deep convolutional network with 8 layers'
    }
}

BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

class FingerprintAnalyzer:
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
    
    def load_and_preprocess_image(self, image_path, target_size):
        image_path = self.clean_path(image_path)
        print(f"Attempting to load image from: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            return np.expand_dims(img, axis=0) / 255.0
        except Exception as e:
            raise ValueError(f"Error processing image at {image_path}: {str(e)}")
    
    def convert_to_functional(self, model, input_shape, model_name):
        """Convert Sequential model to Functional model with unique layer names."""
        try:
            # Create new input with unique name
            inputs = tf.keras.Input(shape=input_shape, name=f'{model_name}_input')
            
            # Build functional model with unique layer names
            x = inputs
            for i, layer in enumerate(model.layers):
                # Create a copy of the layer with a unique name
                config = layer.get_config()
                config['name'] = f'{model_name}_layer_{i}_{config.get("name", "layer")}'
                new_layer = type(layer).from_config(config)
                x = new_layer(x)
            
            functional_model = tf.keras.Model(inputs=inputs, outputs=x, name=f'{model_name}_functional')
            
            # Build by calling once
            sample = np.zeros((1,) + input_shape, dtype=np.float32)
            _ = functional_model(sample)
            
            return functional_model
        except Exception as e:
            print(f"Error converting to functional: {str(e)}")
            return model
    
    def load_models(self):
        print("Loading models...")
        for model_name, config in MODEL_CONFIGS.items():
            try:
                model_path = self.clean_path(config['path'])
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path, compile=False)
                    
                    # Convert Sequential to Functional if needed
                    if isinstance(model, tf.keras.Sequential):
                        print(f"{model_name}: Converting Sequential to Functional...")
                        model = self.convert_to_functional(model, config['input_size'] + (3,), model_name)
                    
                    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    self.models[model_name] = {'model': model, 'config': config}
                    print(f"Loaded {model_name}")
                else:
                    print(f"Model not found: {model_path}")
            except Exception as e:
                print(f"Error loading {model_name}: {str(e)}")
    
    def predict_blood_group(self, model_name, image):
        model_info = self.models[model_name]
        model = model_info['model']
        
        try:
            predictions = model.predict(image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            return {
                'predicted_class': int(predicted_class),
                'predicted_blood_group': BLOOD_GROUPS[predicted_class],
                'confidence': float(confidence),
                'all_probabilities': [float(p) for p in predictions[0]]
            }
        except Exception as e:
            print(f"Error predicting with {model_name}: {str(e)}")
            return None
    
    def find_last_conv_layer(self, model):
        """Find the last convolutional layer in the model."""
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() and ('Conv' in str(type(layer)) or 'conv' in str(type(layer)).lower()):
                return layer.name
        return None
    
    def generate_gradcam(self, model, image, class_index):
        """Generate Grad-CAM heatmap using TensorFlow operations."""
        try:
            last_conv_layer_name = self.find_last_conv_layer(model)
            if last_conv_layer_name is None:
                print("No convolutional layer found")
                return None
            
            # Create gradient model
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(last_conv_layer_name).output, model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                loss = predictions[:, class_index]
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Generate heatmap
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
            heatmap = heatmap.numpy()
            
            # Resize and colorize
            heatmap = cv2.resize(heatmap, (256, 256))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            return heatmap
            
        except Exception as e:
            print(f"Grad-CAM generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_smoothgrad(self, model, image, class_index, n_samples=5, noise_level=0.1):
        """Generate SmoothGrad visualization."""
        try:
            # Ensure image is float32 tensor
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            
            # Generate noisy samples and compute gradients
            gradients = []
            for _ in range(n_samples):
                # Generate noise as float32
                noise = tf.random.normal(shape=tf.shape(image), stddev=noise_level, dtype=tf.float32)
                noisy_image = image + noise
                noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
                
                with tf.GradientTape() as tape:
                    tape.watch(noisy_image)
                    predictions = model(noisy_image)
                    loss = predictions[:, class_index]
                
                grad = tape.gradient(loss, noisy_image)
                gradients.append(grad)
            
            # Average gradients
            avg_gradient = tf.reduce_mean(gradients, axis=0)
            avg_gradient = tf.reduce_mean(tf.abs(avg_gradient), axis=-1)
            avg_gradient = avg_gradient.numpy()[0]
            
            # Normalize and colorize
            avg_gradient = (avg_gradient - avg_gradient.min()) / (avg_gradient.max() - avg_gradient.min() + 1e-8)
            avg_gradient = cv2.resize(avg_gradient, (256, 256))
            avg_gradient = np.uint8(255 * avg_gradient)
            avg_gradient = cv2.applyColorMap(avg_gradient, cv2.COLORMAP_VIRIDIS)
            avg_gradient = cv2.cvtColor(avg_gradient, cv2.COLOR_BGR2RGB)
            
            return avg_gradient
            
        except Exception as e:
            print(f"SmoothGrad generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_occlusion(self, model, image, class_index, patch_size=20):
        """Generate occlusion sensitivity map."""
        try:
            from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
            explainer = OcclusionSensitivity()
            result = explainer.explain(
                validation_data=(image, None),
                model=model,
                class_index=class_index,
                patch_size=patch_size
            )
            if result is not None:
                result = cv2.resize(result, (256, 256))
            return result
        except Exception as e:
            print(f"Occlusion generation failed: {str(e)}")
            return None
    
    def generate_xai_visualizations(self, model_name, image, class_index):
        model_info = self.models[model_name]
        model = model_info['model']
        
        visualizations = {}
        
        # Grad-CAM
        try:
            gradcam = self.generate_gradcam(model, image, class_index)
            if gradcam is not None:
                visualizations['gradcam'] = gradcam
                print(f"{model_name}: Grad-CAM successful")
        except Exception as e:
            print(f"{model_name}: Grad-CAM failed: {str(e)}")
        
        # Occlusion Sensitivity
        try:
            occlusion = self.generate_occlusion(model, image, class_index)
            if occlusion is not None:
                visualizations['occlusion'] = occlusion
                print(f"{model_name}: Occlusion Sensitivity successful")
        except Exception as e:
            print(f"{model_name}: Occlusion failed: {str(e)}")
        
        # SmoothGrad
        try:
            smoothgrad = self.generate_smoothgrad(model, image, class_index)
            if smoothgrad is not None:
                visualizations['smoothgrad'] = smoothgrad
                print(f"{model_name}: SmoothGrad successful")
        except Exception as e:
            print(f"{model_name}: SmoothGrad failed: {str(e)}")
        
        return visualizations if visualizations else None
    
    def analyze_fingerprint(self, image_path):
        print(f"\nAnalyzing fingerprint: {image_path}")
        self.image_path = image_path
        self.load_models()
        
        for model_name, model_info in self.models.items():
            config = model_info['config']
            print(f"\nProcessing {model_name}...")
            
            try:
                image = self.load_and_preprocess_image(
                    image_path, 
                    config['input_size']
                )
                
                prediction = self.predict_blood_group(model_name, image)
                if prediction:
                    self.prediction_results[model_name] = prediction
                    
                    xai_viz = self.generate_xai_visualizations(
                        model_name, image, prediction['predicted_class']
                    )
                    if xai_viz:
                        self.xai_results[model_name] = xai_viz
            except Exception as e:
                print(f"Error processing {model_name}: {str(e)}")
                continue
    
    def create_visualization_grid(self, image_path, model_name):
        try:
            original_img = cv2.imread(image_path)
            if original_img is None:
                raise ValueError(f"Could not load image at {image_path}")
                
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            display_img = cv2.resize(original_img, (256, 256))
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle(f'Fingerprint Analysis - {model_name}', fontsize=16, fontweight='bold')
            
            axes[0, 0].imshow(display_img)
            axes[0, 0].set_title('Original Fingerprint')
            axes[0, 0].axis('off')
            
            def show_or_na(ax, img, title):
                if img is not None:
                    ax.imshow(img)
                    ax.set_title(title)
                else:
                    ax.text(0.5, 0.5, f'{title}\nNot Available', 
                           ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
            
            grad_img = self.xai_results.get(model_name, {}).get('gradcam')
            show_or_na(axes[0, 1], grad_img, 'Grad-CAM\n(Important Regions)')
            
            occ_img = self.xai_results.get(model_name, {}).get('occlusion')
            show_or_na(axes[1, 0], occ_img, 'Occlusion Sensitivity\n(Critical Areas)')
            
            smooth_img = self.xai_results.get(model_name, {}).get('smoothgrad')
            show_or_na(axes[1, 1], smooth_img, 'SmoothGrad\n(Gradient Analysis)')
            
            if model_name in self.prediction_results:
                result = self.prediction_results[model_name]
                fig.text(0.02, 0.98, 
                        f"Prediction: {result['predicted_blood_group']} ({result['confidence']*100:.1f}%)", 
                        transform=fig.transFigure, ha='left', va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating visualization for {model_name}: {str(e)}")
            return None
    
    def get_consensus_prediction(self):
        if not self.prediction_results:
            return None
        
        votes = {}
        for result in self.prediction_results.values():
            blood_group = result['predicted_blood_group']
            votes[blood_group] = votes.get(blood_group, 0) + 1
        
        consensus = max(votes.items(), key=lambda x: x[1])
        avg_confidence = np.mean([r['confidence'] for r in self.prediction_results.values()])
        
        return {
            'blood_group': consensus[0],
            'votes': consensus[1],
            'average_confidence': float(avg_confidence)
        }
    
    def save_visualizations(self):
        output_dir = 'fingerprint_analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name in self.models.keys():
            fig = self.create_visualization_grid(self.image_path, model_name)
            if fig:
                fig.savefig(os.path.join(output_dir, f'{model_name}_analysis.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
    
    def generate_html_report(self):
        consensus = self.get_consensus_prediction()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fingerprint Blood Group Analysis Report</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .model-result {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .high-confidence {{ border-left: 5px solid #27ae60; }}
                .medium-confidence {{ border-left: 5px solid #f39c12; }}
                .low-confidence {{ border-left: 5px solid #e74c3c; }}
                .consensus {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fingerprint Blood Group Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Analysis Summary</h2>
                <div class="consensus">
                    <h3>Consensus Prediction: {consensus['blood_group'] if consensus else 'N/A'}</h3>
                    <p>Models agreeing: {consensus['votes'] if consensus else 0}/4</p>
                    <p>Average confidence: {consensus['average_confidence']*100:.1f}%</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Individual Model Results</h2>
        """
        
        for model_name, result in self.prediction_results.items():
            confidence_class = 'high-confidence' if result['confidence'] > 0.8 else 'medium-confidence' if result['confidence'] > 0.6 else 'low-confidence'
            
            html_content += f"""
                <div class="model-result {confidence_class}">
                    <h3>{model_name}</h3>
                    <p><strong>Predicted Blood Group:</strong> {result['predicted_blood_group']}</p>
                    <p><strong>Confidence:</strong> {result['confidence']*100:.1f}%</p>
                    <p><strong>Model Description:</strong> {MODEL_CONFIGS[model_name]['description']}</p>
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Visual Analysis</h2>
                <p>Visualization files have been saved in the 'fingerprint_analysis_results' directory</p>
            </div>
        </body>
        </html>
        """
        
        report_path = 'fingerprint_analysis_results/analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def generate_complete_report(self, image_path):
        print("Starting Fingerprint Blood Group Analysis...")
        print("=" * 60)
        
        image_path = self.clean_path(image_path)
        self.analyze_fingerprint(image_path)
        
        if not self.prediction_results:
            print("No successful predictions made.")
            return None
        
        consensus = self.get_consensus_prediction()
        
        print("\nANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Consensus Blood Group: {consensus['blood_group']}")
        print(f"Models agreeing: {consensus['votes']}/4")
        print(f"Average confidence: {consensus['average_confidence']*100:.1f}%")
        
        print("\nINDIVIDUAL MODEL RESULTS")
        print("-" * 60)
        for model_name, result in self.prediction_results.items():
            print(f"{model_name:12}: {result['predicted_blood_group']:4} ({result['confidence']*100:5.1f}%)")
        
        print("\nSaving visualizations...")
        self.save_visualizations()
        
        print("Generating HTML report...")
        report_path = self.generate_html_report()
        
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'consensus': consensus,
            'individual_results': self.prediction_results
        }
        
        with open('fingerprint_analysis_results/analysis_data.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"\nAnalysis complete!")
        print(f"Report saved to: {report_path}")
        
        try:
            webbrowser.open('file://' + os.path.abspath(report_path))
            print("Report opened in browser")
        except:
            print("Could not open report in browser")
        
        return consensus

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    analyzer = FingerprintAnalyzer()
    
    print("Fingerprint Blood Group Analysis System")
    print("=" * 50)
    
    default_image_path = 'dataset/test/fingerprint_sample.BMP'
    
    if os.path.exists(default_image_path):
        print(f"Using default image: {default_image_path}")
        image_path = default_image_path
    else:
        image_path = input("Enter path to your fingerprint image: ").strip()
        if not image_path:
            print("No image path provided. Exiting.")
            return
    
    image_path = analyzer.clean_path(image_path)
    print(f"Final image path: {image_path}")
    
    try:
        result = analyzer.generate_complete_report(image_path)
        
        if result:
            print(f"\nFINAL PREDICTION: {result['blood_group']}")
            print(f"Confidence: {result['average_confidence']*100:.1f}%")
            print(f"Model agreement: {result['votes']}/4")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()