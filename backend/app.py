#!/usr/bin/env python3
"""
ExoNeural Backend API
NASA Space Apps Challenge 2024
Team: ExoNeural

Flask API for exoplanet detection predictions
"""

from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
import random
import numpy as np # type: ignore
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class ExoplanetPredictor:
    """
    Dummy ML model simulator for exoplanet detection
    In production, this would load your trained model
    """
    
    def __init__(self):
        self.model_loaded = True
        logger.info("ExoNeural model initialized")
    
    def predict(self, orbital_period, radius, duration):
        """
        Simulate exoplanet prediction based on input parameters
        
        Args:
            orbital_period (float): Orbital period in days
            radius (float): Planet radius in Earth radii
            duration (float): Transit duration in hours
            
        Returns:
            dict: Prediction result with classification and confidence
        """
        try:
            # Simulate model processing time
            import time
            time.sleep(0.5)
            
            # Simple heuristic for demo purposes
            # In reality, this would be your trained ML model
            score = 0.0
            
            # Orbital period influence (Earth-like periods get higher scores)
            if 200 <= orbital_period <= 400:
                score += 0.4
            elif 50 <= orbital_period <= 600:
                score += 0.2
            
            # Radius influence (Earth to Neptune size)
            if 0.5 <= radius <= 4.0:
                score += 0.3
            elif radius <= 10.0:
                score += 0.1
            
            # Duration influence
            if 2 <= duration <= 12:
                score += 0.3
            
            # Add some randomness to simulate model uncertainty
            score += random.uniform(-0.2, 0.2)
            score = max(0.1, min(0.99, score))  # Clamp between 0.1 and 0.99
            
            # Determine classification
            if score >= 0.7:
                classification = "Confirmed Exoplanet"
            elif score >= 0.4:
                classification = "Candidate Exoplanet"
            else:
                classification = "False Positive"
            
            return {
                "prediction": classification,
                "confidence": round(score, 3)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "prediction": "Error",
                "confidence": 0.0,
                "error": str(e)
            }

# Initialize the predictor
predictor = ExoplanetPredictor()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "ExoNeural API",
        "version": "1.0.0",
        "team": "ExoNeural Team - NASA Space Apps Challenge 2024"
    })

@app.route('/predict', methods=['POST'])
def predict_exoplanet():
    """
    Main prediction endpoint
    Expects JSON with orbital_period, radius, and duration
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON",
                "status": "error"
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['orbital_period', 'radius', 'duration']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "status": "error"
            }), 400
        
        # Extract and validate parameters
        try:
            orbital_period = float(data['orbital_period'])
            radius = float(data['radius'])
            duration = float(data['duration'])
        except (ValueError, TypeError):
            return jsonify({
                "error": "All parameters must be numeric",
                "status": "error"
            }), 400
        
        # Validate parameter ranges
        if orbital_period <= 0 or radius <= 0 or duration <= 0:
            return jsonify({
                "error": "All parameters must be positive",
                "status": "error"
            }), 400
        
        logger.info(f"Processing prediction request: period={orbital_period}, radius={radius}, duration={duration}")
        
        # Make prediction
        result = predictor.predict(orbital_period, radius, duration)
        
        # Add metadata
        result.update({
            "status": "success",
            "parameters": {
                "orbital_period": orbital_period,
                "radius": radius,
                "duration": duration
            },
            "model_version": "ExoNeural-v1.0",
            "timestamp": str(np.datetime64('now'))
        })
        
        logger.info(f"Prediction completed: {result['prediction']} (confidence: {result['confidence']})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "status": "error",
            "details": str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for CSV data
    Expects JSON with array of parameter objects
    """
    try:
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON",
                "status": "error"
            }), 400
        
        data = request.get_json()
        
        if 'data' not in data or not isinstance(data['data'], list):
            return jsonify({
                "error": "Request must contain 'data' array",
                "status": "error"
            }), 400
        
        results = []
        for i, item in enumerate(data['data']):
            try:
                orbital_period = float(item['orbital_period'])
                radius = float(item['radius'])
                duration = float(item['duration'])
                
                prediction = predictor.predict(orbital_period, radius, duration)
                prediction['row_index'] = i
                results.append(prediction)
                
            except Exception as e:
                results.append({
                    "row_index": i,
                    "prediction": "Error",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        return jsonify({
            "status": "success",
            "results": results,
            "total_processed": len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "status": "error"
        }), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)