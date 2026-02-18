import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any
from model_utils import preprocess_image_for_model

class EnsemblePredictor:
    """
    Ensemble predictor that combines predictions from multiple models.
    Handles different model types and input requirements.
    """
    
    def __init__(self, models: Dict[str, Tuple[tf.keras.Model, str]]):
        """
        Initialize with a dictionary of models and their types.
        
        Args:
            models: Dictionary mapping model names to (model, model_type) tuples
        """
        self.models = models
        self.class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        
    def predict_single_model(
        self, 
        model: tf.keras.Model, 
        model_type: str, 
        image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Make prediction using a single model.
        
        Args:
            model: The Keras model to use for prediction
            model_type: Type of the model ('alexnet', 'lenet', 'resnet', 'vgg')
            image: Input image in RGB format (H, W, 3)
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess image for the specific model
            img = preprocess_image_for_model(image, model_type)
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            # Make prediction
            pred = model.predict(img, verbose=0)
            
            # Handle different prediction output formats
            if isinstance(pred, list):
                pred = pred[0]  # Take first output if multiple outputs
            pred = np.array(pred).flatten()  # Ensure it's a flat array
            
            # Get top prediction
            pred_class = np.argmax(pred)
            confidence = float(pred[pred_class])
            
            # Get top 3 predictions
            top_k = 3
            top_indices = np.argsort(pred)[::-1][:top_k]
            top_features = [self.class_names[i] for i in top_indices]
            
            return {
                'prediction': self.class_names[pred_class],
                'confidence': confidence,
                'probabilities': pred.tolist(),
                'top_features': top_features,
                'success': True
            }
            
        except Exception as e:
            print(f"Error in model prediction: {str(e)}")
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'probabilities': [1.0/len(self.class_names)] * len(self.class_names),
                'top_features': [],
                'success': False,
                'error': str(e)
            }
    
    def predict_ensemble(
        self, 
        image: np.ndarray,
        method: str = 'weighted_voting'
    ) -> Dict[str, Any]:
        """
        Make prediction using all models in the ensemble.
        
        Args:
            image: Input image in RGB format (H, W, 3)
            method: Ensemble method ('weighted_voting' or 'averaging')
            
        Returns:
            Dictionary containing ensemble prediction results
        """
        predictions = {}
        all_confidences = []
        successful_models = 0
        
        # Get predictions from all models
        for model_name, (model, model_type) in self.models.items():
            result = self.predict_single_model(model, model_type, image)
            if result['success']:
                predictions[model_name] = result
                all_confidences.append(result['confidence'])
                successful_models += 1
        
        if successful_models == 0:
            raise ValueError("No models made successful predictions")
        
        # Different ensemble methods
        if method == 'weighted_voting':
            return self._weighted_voting(predictions)
        elif method == 'averaging':
            return self._probability_averaging(predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def _weighted_voting(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Weighted voting ensemble method."""
        weighted_votes = {}
        
        # Calculate weighted votes
        for model_name, pred in predictions.items():
            pred_class = pred['prediction']
            confidence = pred['confidence']
            
            if pred_class not in weighted_votes:
                weighted_votes[pred_class] = 0.0
            weighted_votes[pred_class] += confidence
        
        # Get the prediction with highest total confidence
        if weighted_votes:
            ensemble_pred = max(weighted_votes.items(), key=lambda x: x[1])[0]
            total_confidence = weighted_votes[ensemble_pred]
            avg_confidence = total_confidence / sum(1 for p in predictions.values() 
                                                  if p['prediction'] == ensemble_pred)
        else:
            ensemble_pred = 'Unknown'
            avg_confidence = 0.0
        
        # Calculate agreement
        agreement = sum(1 for p in predictions.values() 
                       if p['prediction'] == ensemble_pred)
        
        return {
            'ensemble_prediction': ensemble_pred,
            'ensemble_confidence': avg_confidence,
            'agreement_count': agreement,
            'total_models': len(predictions),
            'all_predictions': predictions,
            'method': 'weighted_voting'
        }
    
    def _probability_averaging(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Probability averaging ensemble method."""
        # Initialize average probabilities
        avg_probs = np.zeros(len(self.class_names))
        
        # Sum probabilities from all models
        for pred in predictions.values():
            probs = pred['probabilities'][:len(self.class_names)]  # Ensure correct length
            avg_probs += np.array(probs)
        
        # Calculate average
        avg_probs /= len(predictions)
        
        # Get final prediction
        pred_class = np.argmax(avg_probs)
        confidence = float(avg_probs[pred_class])
        
        return {
            'ensemble_prediction': self.class_names[pred_class],
            'ensemble_confidence': confidence,
            'average_probabilities': avg_probs.tolist(),
            'all_predictions': predictions,
            'method': 'probability_averaging'
        }
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return performance metrics for all models.
        
        Returns:
            Dictionary containing various performance metrics
        """
        metrics = {}
        for model_name, (model, _) in self.models.items():
            try:
                # Get model summary as string
                stringlist = []
                model.summary(print_fn=lambda x: stringlist.append(x))
                summary = "\n".join(stringlist)
                
                metrics[model_name] = {
                    'trainable_params': model.count_params(),
                    'layers': len(model.layers),
                    'input_shape': model.input_shape,
                    'output_shape': model.output_shape,
                    'summary': summary
                }
            except Exception as e:
                metrics[model_name] = {
                    'error': f"Could not get metrics: {str(e)}"
                }
        
        return metrics

# Helper function for backward compatibility
def predict_ensemble(
    models: Dict[str, Tuple[tf.keras.Model, str]],
    image: np.ndarray,
    class_names: List[str],
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper function for backward compatibility.
    """
    ensemble = EnsemblePredictor(models)
    ensemble.class_names = class_names
    return ensemble.predict_ensemble(image, **kwargs)