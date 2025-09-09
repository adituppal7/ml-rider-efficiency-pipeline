from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
import io
import uvicorn
import asyncio
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import tempfile
import httpx
import json
import boto3
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, LinearRegression, Lasso

# Import optional ML libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class R2StorageManager:
    """Manages Cloudflare R2 storage operations."""
    
    def __init__(self, account_id: str, access_key: str, secret_key: str, bucket_name: str):
        self.bucket_name = bucket_name
        self.endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='auto'
        )
    
    async def upload_file(self, file_content: bytes, filename: str) -> Dict[str, str]:
        """Upload file to R2 storage."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=unique_filename,
                Body=file_content,
                ContentType='application/octet-stream'
            )
            
            logger.info(f"Uploaded {filename} to R2 as {unique_filename}")
            return {"filename": unique_filename, "status": "success"}
            
        except Exception as e:
            logger.error(f"Failed to upload {filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    async def download_file(self, filename: str) -> bytes:
        """Download file from R2 storage."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
            return response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise HTTPException(status_code=404, detail=f"File {filename} not found")
            else:
                raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")
    
    async def list_files(self) -> List[str]:
        """List training files in R2 bucket."""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            files = []
            for obj in response.get('Contents', []):
                if obj['Key'].lower().endswith(('.csv', '.xlsx', '.xls')):
                    files.append(obj['Key'])
            return files
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []

class TursoManager:
    """Manages Turso database operations via HTTP API."""
    
    def __init__(self, database_url: str, auth_token: str):
        self.database_url = database_url.rstrip('/')
        self.auth_token = auth_token
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
    
    async def execute_query(self, query: str, params: list = None) -> dict:
        """Execute SQL query via Turso HTTP API."""
        try:
            payload = {
                "statements": [
                    {
                        "q": query,
                        "params": params or []
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.database_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Turso query failed: {response.status_code} - {response.text}")
                    raise HTTPException(status_code=500, detail=f"Database query failed: {response.text}")
                    
        except Exception as e:
            logger.error(f"Turso query error: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def save_analysis(self, analysis_data: dict) -> str:
        """Save analysis results to database."""
        try:
            query = """
            INSERT INTO analysis_results 
            (filename, predicted_range, efficiency_score, model_type, features_analyzed, 
             data_points, throttle_avg, soc_start, soc_end, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """
            
            # Convert numpy types to Python types for JSON serialization
            def safe_convert(value):
                if value is None:
                    return None
                if hasattr(value, 'item'):  # NumPy scalar
                    return value.item()
                return value
    
            params = [
                safe_convert(analysis_data.get('filename')),
                safe_convert(analysis_data.get('predicted_range')),
                safe_convert(analysis_data.get('efficiency_score')),
                safe_convert(analysis_data.get('model_type')),
                safe_convert(analysis_data.get('features_analyzed')),
                safe_convert(analysis_data.get('data_points')),
                safe_convert(analysis_data.get('throttle_avg')),
                safe_convert(analysis_data.get('soc_start')),
                safe_convert(analysis_data.get('soc_end'))
            ]
            
            result = await self.execute_query(query, params)
            
            # Generate a simple timestamp-based ID for logging/reference
            import time
            analysis_id = str(int(time.time()))
            logger.info(f"Successfully saved analysis with reference ID: {analysis_id}")
            return analysis_id
                    
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            raise HTTPException(status_code=500, detail="Failed to save analysis")   
                        
    async def create_tables(self):
        """Create tables if they don't exist."""
        try:
            # Create analysis_results table
            analysis_table_query = """
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                predicted_range REAL,
                efficiency_score REAL,
                model_type TEXT,
                features_analyzed INTEGER,
                data_points INTEGER,
                throttle_avg REAL,
                soc_start REAL,
                soc_end REAL,
                created_at TEXT
            )
            """
            
            await self.execute_query(analysis_table_query)
            
            # Create training_history table
            training_table_query = """
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT,
                samples_count INTEGER,
                features_count INTEGER,
                test_r2 REAL,
                test_mae REAL,
                cv_score REAL,
                files_processed INTEGER,
                overfitting_gap REAL,
                train_r2 REAL,
                train_mae REAL,
                created_at TEXT
            )
            """
            
            await self.execute_query(training_table_query)
            logger.info("Tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")

class AdvancedRiderEfficiencyScorer:
    """ML model with Turso integration and consistent feature handling."""

    def __init__(self, model_path: str = "efficiency_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.model_type = None
        self.feature_names = None
        
        # EXACT feature order from your trained scaler (41 features)
        self.SCALER_FEATURES = [
            'avg_throttle', 'median_throttle', 'max_throttle', 'std_throttle', 'throttle_range',
            'throttle_p25', 'throttle_p75', 'throttle_p90', 'throttle_p95', 'throttle_cv',
            'low_throttle_ratio', 'moderate_throttle_ratio', 'high_throttle_ratio',
            'eco_events', 'moderate_events', 'high_events', 'max_sustained_eco',
            'max_sustained_moderate', 'max_sustained_high', 'avg_throttle_change',
            'max_throttle_change', 'throttle_smoothness', 'sudden_changes',
            'avg_current', 'max_current', 'std_current', 'current_p95', 'high_current_ratio',
            'current_efficiency', 'avg_temp_delta', 'temp_imbalance_ratio', 'temp_stability',
            'start_soc', 'end_soc', 'soc_consumed', 'avg_soc', 'min_soc', 'soc_efficiency',
            'soc_range_ratio', 'soc_drain_rate', 'soc_consistency'
        ]
        
        # Final 10 features selected by your model
        self.SELECTED_FEATURES = [
            'throttle_p75', 'throttle_p90', 'throttle_p95', 'avg_current', 'max_current',
            'std_current', 'current_p95', 'temp_stability', 'soc_drain_rate', 'soc_consistency'
        ]
        
        if os.path.exists(self.model_path):
            self._load_model()

    def _load_model(self):
        """Load trained model."""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data.get('feature_selector')
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            logger.info(f"Loaded {self.model_type} model with {len(self.feature_names)} features")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def extract_features(self, throttle_data: List[float],
                        current_data: Optional[List[float]] = None,
                        cell_temp_delta_data: Optional[List[float]] = None,
                        soc_data: Optional[List[float]] = None) -> Dict:
        """Extract features with exact same logic as Colab training."""
        
        throttle = np.array([x for x in throttle_data if pd.notna(x) and 0 <= x <= 100])
        
        if len(throttle) == 0:
            return self._get_zero_features()

        features = {}
        
        # Basic throttle statistics
        features['avg_throttle'] = np.mean(throttle)
        features['median_throttle'] = np.median(throttle)
        features['max_throttle'] = np.max(throttle)
        features['min_throttle'] = np.min(throttle)
        features['std_throttle'] = np.std(throttle)
        features['throttle_range'] = features['max_throttle'] - features['min_throttle']

        # Percentiles
        features['throttle_p25'] = np.percentile(throttle, 25)
        features['throttle_p75'] = np.percentile(throttle, 75)
        features['throttle_p90'] = np.percentile(throttle, 90)
        features['throttle_p95'] = np.percentile(throttle, 95)

        # Coefficient of variation
        features['throttle_cv'] = features['std_throttle'] / (features['avg_throttle'] + 1e-6)

        # Efficiency ratios
        features['low_throttle_ratio'] = np.mean(throttle <= 20)
        features['moderate_throttle_ratio'] = np.mean((throttle > 20) & (throttle <= 60))
        features['high_throttle_ratio'] = np.mean((throttle > 60) & (throttle <= 80))
        features['aggressive_throttle_ratio'] = np.mean(throttle > 80)

        # Event counts
        features['eco_events'] = np.sum(throttle <= 20)
        features['moderate_events'] = np.sum((throttle > 20) & (throttle <= 60))
        features['high_events'] = np.sum((throttle > 60) & (throttle <= 80))
        features['aggressive_events'] = np.sum(throttle > 80)

        # Sustained behavior
        features['max_sustained_eco'] = self._max_sustained_condition(throttle <= 20)
        features['max_sustained_moderate'] = self._max_sustained_condition((throttle > 20) & (throttle <= 60))
        features['max_sustained_high'] = self._max_sustained_condition(throttle > 60)
        features['max_sustained_aggressive'] = self._max_sustained_condition(throttle > 80)

        # Throttle changes
        if len(throttle) > 1:
            throttle_diff = np.diff(throttle)
            features['avg_throttle_change'] = np.mean(np.abs(throttle_diff))
            features['max_throttle_change'] = np.max(np.abs(throttle_diff))
            features['throttle_smoothness'] = 1 / (1 + features['avg_throttle_change'])
            features['sudden_changes'] = np.sum(np.abs(throttle_diff) > 10)
        else:
            features['avg_throttle_change'] = 0
            features['max_throttle_change'] = 0
            features['throttle_smoothness'] = 1
            features['sudden_changes'] = 0

        # Current features
        if current_data:
            current = np.array([x for x in current_data if pd.notna(x) and x >= 0])
            if len(current) > 0:
                features['avg_current'] = np.mean(current)
                features['max_current'] = np.max(current)
                features['std_current'] = np.std(current)
                features['current_p95'] = np.percentile(current, 95)
                features['high_current_ratio'] = np.mean(current > np.percentile(current, 80))
                features['current_efficiency'] = features['avg_throttle'] / (features['avg_current'] + 1e-6)
            else:
                features.update(self._get_zero_current_features())
        else:
            features.update(self._get_zero_current_features())

        # Temperature features
        if cell_temp_delta_data:
            temp_delta = np.array([x for x in cell_temp_delta_data if pd.notna(x)])
            if len(temp_delta) > 0:
                features['avg_temp_delta'] = np.mean(temp_delta)
                features['max_temp_delta'] = np.max(temp_delta)
                features['temp_imbalance_ratio'] = features['max_temp_delta'] / (features['avg_temp_delta'] + 1e-6)
                features['temp_stability'] = 1 / (1 + np.std(temp_delta))
            else:
                features.update(self._get_zero_temp_features())
        else:
            features.update(self._get_zero_temp_features())

        # SOC features
        if soc_data:
            soc = np.array([x for x in soc_data if pd.notna(x) and 0 <= x <= 100])
            if len(soc) > 0:
                features['start_soc'] = soc[0]
                features['end_soc'] = soc[-1]
                features['soc_consumed'] = features['start_soc'] - features['end_soc']
                features['avg_soc'] = np.mean(soc)
                features['min_soc'] = np.min(soc)
                features['soc_efficiency'] = features['soc_consumed'] / (features['avg_throttle'] + 1e-6)
                features['soc_range_ratio'] = features['soc_consumed'] / (features['avg_throttle'] + 1e-6)

                if len(soc) > 1:
                    soc_diff = np.diff(soc)
                    features['soc_drain_rate'] = np.mean(np.abs(soc_diff))
                    features['soc_consistency'] = 1 / (1 + np.std(soc_diff))
                else:
                    features['soc_drain_rate'] = 0
                    features['soc_consistency'] = 1
            else:
                features.update(self._get_zero_soc_features())
        else:
            features.update(self._get_zero_soc_features())

        return features

    def _max_sustained_condition(self, condition_array: np.ndarray) -> int:
        """Calculate maximum sustained duration."""
        max_duration = 0
        current_duration = 0
        for condition in condition_array:
            if condition:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        return max_duration

    def _get_zero_features(self) -> Dict:
        """Return zero-valued features for empty data."""
        return {
            'avg_throttle': 0, 'median_throttle': 0, 'max_throttle': 0, 'min_throttle': 0,
            'std_throttle': 0, 'throttle_range': 0, 'throttle_p25': 0, 'throttle_p75': 0,
            'throttle_p90': 0, 'throttle_p95': 0, 'throttle_cv': 0,
            'low_throttle_ratio': 0, 'moderate_throttle_ratio': 0, 'high_throttle_ratio': 0,
            'aggressive_throttle_ratio': 0, 'eco_events': 0, 'moderate_events': 0,
            'high_events': 0, 'aggressive_events': 0, 'max_sustained_eco': 0,
            'max_sustained_moderate': 0, 'max_sustained_high': 0, 'max_sustained_aggressive': 0,
            'avg_throttle_change': 0, 'max_throttle_change': 0, 'throttle_smoothness': 1,
            'sudden_changes': 0, **self._get_zero_current_features(), 
            **self._get_zero_temp_features(), **self._get_zero_soc_features()
        }

    def _get_zero_current_features(self) -> Dict:
        return {
            'avg_current': 0, 'max_current': 0, 'std_current': 0,
            'current_p95': 0, 'high_current_ratio': 0, 'current_efficiency': 0
        }

    def _get_zero_temp_features(self) -> Dict:
        return {
            'avg_temp_delta': 0, 'max_temp_delta': 0, 'temp_imbalance_ratio': 0, 'temp_stability': 1
        }

    def _get_zero_soc_features(self) -> Dict:
        return {
            'start_soc': 0, 'end_soc': 0, 'soc_consumed': 0, 'avg_soc': 0, 'min_soc': 0,
            'soc_efficiency': 0, 'soc_range_ratio': 0, 'soc_drain_rate': 0, 'soc_consistency': 1
        }

    def calculate_efficiency_score(self, predicted_range, throttle_data=None, soc_data=None):
        """Calculate efficiency score primarily based on range performance with driving behavior bonus."""
        if throttle_data is None or soc_data is None:
            # Fallback to range-based score
            return max(0, min(100, round(predicted_range / 1.2, 1)))

        # Calculate efficiency based on multiple factors with range as primary
        scores = []

        # 1. RANGE PERFORMANCE (60% weight) - Primary factor
        if predicted_range >= 120:
            range_score = 100  # Excellent range
        elif predicted_range >= 100:
            range_score = 85 + (predicted_range - 100) * 0.75
        elif predicted_range >= 80:
            range_score = 70 + (predicted_range - 80) * 0.75
        elif predicted_range >= 60:
            range_score = 50 + (predicted_range - 60) * 1.0
        elif predicted_range >= 40:
            range_score = 30 + (predicted_range - 40) * 1.0
        else:
            range_score = predicted_range * 0.75

        range_score = range_score * 0.6
        scores.append(range_score)

        # 2. SOC EFFICIENCY (25% weight) - Battery usage efficiency
        soc_scores = []
        soc = np.array([x for x in soc_data if pd.notna(x) and 0 <= x <= 100])
        if len(soc) > 1:
            soc_consumed = soc[0] - soc[-1]
            if soc_consumed > 0:
                range_per_soc = predicted_range / soc_consumed
                if range_per_soc >= 8:
                    soc_efficiency = 100
                elif range_per_soc >= 6:
                    soc_efficiency = 80 + (range_per_soc - 6) * 10
                elif range_per_soc >= 4:
                    soc_efficiency = 60 + (range_per_soc - 4) * 10
                elif range_per_soc >= 2:
                    soc_efficiency = 40 + (range_per_soc - 2) * 10
                else:
                    soc_efficiency = range_per_soc * 20

                soc_scores.append(soc_efficiency)

            soc_variation = np.std(np.diff(soc))
            if soc_variation < 0.5:
                stability_bonus = 10
            elif soc_variation < 1.0:
                stability_bonus = 5
            else:
                stability_bonus = 0

            soc_scores.append(stability_bonus)

        if soc_scores:
            soc_score = np.mean(soc_scores) * 0.25
            scores.append(soc_score)

        # 3. DRIVING BEHAVIOR (15% weight) - Bonus for good driving
        driving_bonus = 0
        throttle = np.array([x for x in throttle_data if pd.notna(x) and 0 <= x <= 100])
        if len(throttle) > 0:
            eco_ratio = np.mean(throttle <= 30)
            if eco_ratio >= 0.8:
                driving_bonus += 10
            elif eco_ratio >= 0.6:
                driving_bonus += 5
            elif eco_ratio >= 0.4:
                driving_bonus += 2

            throttle_std = np.std(throttle)
            if throttle_std < 5:
                driving_bonus += 5
            elif throttle_std < 10:
                driving_bonus += 2

            driving_bonus = min(15, driving_bonus)

        driving_score = driving_bonus * 0.15
        scores.append(driving_score)

        # Calculate final score
        if scores:
            final_score = sum(scores)
            return max(0, min(100, round(final_score, 1)))
        else:
            return max(0, min(100, round(predicted_range / 1.2, 1)))

    async def process_file_and_predict(self, file_content: bytes, filename: str) -> Dict:
        """Process file and predict with proper efficiency scoring."""
        try:
            # Read file
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(io.BytesIO(file_content))
            else:
                df = pd.read_excel(io.BytesIO(file_content))
            
            # EXACT same column processing as Colab
            df.columns = df.columns.str.strip().str.lower()
            
            if 'throttle%' not in df.columns:
                raise ValueError("Missing 'throttle%' column")
            
            throttle_data = pd.to_numeric(df['throttle%'], errors='coerce').dropna().tolist()
            
            current_data = None
            if 'current( a )' in df.columns:
                current_data = pd.to_numeric(df['current( a )'], errors='coerce').dropna().tolist()
            
            temp_data = None
            if 'cell temp delta' in df.columns:
                temp_data = pd.to_numeric(df['cell temp delta'], errors='coerce').dropna().tolist()
            
            # SOC detection - EXACT same logic as Colab
            soc_data = None
            soc_column_names = [
                'soc( % )', 'soc(%)', 'soc %', 'soc%', 'soc',
                'state of charge', 'state of charge %', 'state of charge(%)',
                'battery level', 'battery level %', 'battery level(%)',
                'battery %', 'battery(%)'
            ]
            
            voltage_keywords = ['voltage', 'v)', 'v ', 'cell voltage', 'battery voltage', 'pack_voltage']
            
            found_soc_column = None
            for col in df.columns:
                col_lower = col.lower()
                if any(voltage_key in col_lower for voltage_key in voltage_keywords):
                    continue
                for soc_name in soc_column_names:
                    if soc_name.lower() in col_lower:
                        found_soc_column = col
                        break
                if found_soc_column:
                    break
            
            if found_soc_column:
                soc_data = pd.to_numeric(df[found_soc_column], errors='coerce').dropna().tolist()
            
            if len(throttle_data) < 5:
                raise ValueError("Insufficient throttle data")
            
            # Extract ALL features first
            all_features = self.extract_features(throttle_data, current_data, temp_data, soc_data)
            
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Create feature DataFrame with EXACT scaler features (41 features)
            feature_values = []
            for feature_name in self.SCALER_FEATURES:
                if feature_name in all_features:
                    feature_values.append(all_features[feature_name])
                else:
                    logger.warning(f"Missing feature: {feature_name}, using 0")
                    feature_values.append(0)
            
            # Create DataFrame with exact column names the scaler expects
            feature_df = pd.DataFrame([feature_values], columns=self.SCALER_FEATURES)
            
            logger.info(f"Created feature DataFrame with {len(feature_df.columns)} features for scaler")
            
            # Apply preprocessing pipeline: Scaler -> Feature Selector -> Model
            features_scaled = self.scaler.transform(feature_df)
            logger.info(f"Scaled features shape: {features_scaled.shape}")
            
            # Apply feature selection (reduces 41 -> 10 features)
            if self.feature_selector:
                features_selected = self.feature_selector.transform(features_scaled)
                logger.info(f"Selected features shape: {features_selected.shape}")
            else:
                features_selected = features_scaled
            
            # Make prediction
            prediction = self.model.predict(features_selected)[0]
            
            # Calculate proper efficiency score using the raw data
            efficiency_score = self.calculate_efficiency_score(
                predicted_range=float(prediction),
                throttle_data=throttle_data,
                soc_data=soc_data
            )
            
            return {
                "predicted_range": round(float(prediction), 2),
                "efficiency_score": round(efficiency_score, 1),
                "model_type": self.model_type,
                "features_analyzed": len(feature_df.columns),
                "data_points": len(throttle_data),
                "throttle_avg": all_features.get('avg_throttle', 0),
                "soc_start": all_features.get('start_soc', 0),
                "soc_end": all_features.get('end_soc', 0),
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def build_training_dataset_from_r2(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Build training dataset using EXACT Colab logic."""
        logger.info("Building training dataset from R2 files...")
        
        rows = []
        
        for i, df in enumerate(dataframes):
            filename = f"r2_file_{i+1}"
            logger.info(f"Processing: {filename}")
            
            try:
                # EXACT same column processing as Colab
                df.columns = df.columns.str.strip().str.lower()
                
                # Check required columns
                if 'throttle%' not in df.columns:
                    logger.warning(f"Skipping {filename}: Missing 'throttle%' column")
                    continue
                    
                if 'range' not in df.columns:
                    logger.warning(f"Skipping {filename}: Missing 'range' column")
                    continue
                
                # Extract throttle data
                throttle_data = pd.to_numeric(df['throttle%'], errors='coerce').dropna().tolist()
                
                # Extract current data
                current_data = None
                if 'current( a )' in df.columns:
                    current_data = pd.to_numeric(df['current( a )'], errors='coerce').dropna().tolist()
                
                # Extract temperature data
                temp_data = None
                if 'cell temp delta' in df.columns:
                    temp_data = pd.to_numeric(df['cell temp delta'], errors='coerce').dropna().tolist()
                
                # SOC detection - EXACT same logic as Colab
                soc_data = None
                soc_column_names = [
                    'soc( % )', 'soc(%)', 'soc %', 'soc%', 'soc',
                    'state of charge', 'state of charge %', 'state of charge(%)',
                    'battery level', 'battery level %', 'battery level(%)',
                    'battery %', 'battery(%)'
                ]
                
                voltage_keywords = ['voltage', 'v)', 'v ', 'cell voltage', 'battery voltage', 'pack_voltage']
                
                found_soc_column = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    
                    # Skip voltage-related columns
                    if any(voltage_key in col_lower for voltage_key in voltage_keywords):
                        continue
                    
                    # Check for SOC matches
                    for soc_name in soc_column_names:
                        if soc_name.lower() in col_lower:
                            found_soc_column = col
                            break
                    
                    if found_soc_column:
                        break
                
                if found_soc_column:
                    soc_data = pd.to_numeric(df[found_soc_column], errors='coerce').dropna().tolist()
                    
                    # Validate SOC data quality
                    if len(soc_data) > 0:
                        soc_min, soc_max = min(soc_data), max(soc_data)
                        
                        # Check if values look like SOC (typically 0-100 range)
                        if 0 <= soc_min <= 100 and 0 <= soc_max <= 100:
                            logger.info(f"SOC data found in column '{found_soc_column}': {len(soc_data)} points")
                            
                            soc_range = soc_max - soc_min
                            soc_consumed = soc_data[0] - soc_data[-1] if len(soc_data) > 1 else 0
                            
                            if soc_range < 1:
                                logger.warning(f"Very low SOC variation ({soc_range:.1f}%) - may affect accuracy")
                            if soc_consumed < 0:
                                logger.warning("SOC increased during ride - check data quality")
                        else:
                            logger.warning(f"Column '{found_soc_column}' found but values don't look like SOC")
                            soc_data = None
                            found_soc_column = None
                    else:
                        logger.warning(f"Column '{found_soc_column}' found but no valid numeric data")
                        soc_data = None
                        found_soc_column = None
                else:
                    logger.warning(f"No SOC column found in {filename}")
                
                # Extract range values
                range_values = pd.to_numeric(df['range'], errors='coerce').dropna()
                if len(range_values) == 0:
                    logger.warning(f"Skipping {filename}: No valid range values")
                    continue
                
                range_val = range_values.mean()
                
                # Validation checks
                if len(throttle_data) < 10:
                    logger.warning(f"Skipping {filename}: Insufficient throttle data ({len(throttle_data)} points)")
                    continue
                
                if range_val <= 0 or range_val > 500:
                    logger.warning(f"Skipping {filename}: Invalid range value ({range_val})")
                    continue
                
                # Extract features using the EXACT same method
                features = self.extract_features(throttle_data, current_data, temp_data, soc_data)
                features['range'] = range_val
                rows.append(features)
                logger.info(f"Processed {filename}: {len(throttle_data)} data points, range = {range_val:.1f} km")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        if not rows:
            logger.error("No valid data found!")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(rows)
        training_df = result_df.copy()
        
        logger.info(f"Built training dataset: {len(training_df)} samples, {len(training_df.columns)-1} features")
        
        # Check if SOC features were included
        soc_features = [col for col in training_df.columns if 'soc' in col.lower()]
        if soc_features:
            logger.info(f"SOC features included: {soc_features}")
        else:
            logger.warning("No SOC features found in training data")
        
        return training_df

    def preprocess_data_exact(self, df: pd.DataFrame) -> tuple:
        """EXACT same preprocessing as Colab."""
        logger.info(f"Initial dataset: {len(df)} samples")
        
        # EXACT: Keep all outliers for throttle spike detection
        logger.info("Keeping all data points including outliers (for throttle spike analysis)")
        df_clean = df.copy()
        
        # Separate features and target
        feature_cols = [col for col in df_clean.columns if col != 'range']
        X = df_clean[feature_cols].copy()
        y = df_clean['range'].copy()
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Remove features with zero variance
        zero_var_cols = X.columns[X.var() == 0].tolist()
        if zero_var_cols:
            logger.info(f"Removing zero-variance features: {zero_var_cols}")
            X = X.drop(columns=zero_var_cols)
        
        logger.info(f"Final dataset: {len(X)} samples, {len(X.columns)} features")
        return X, y

    def train_model_exact(self, training_df: pd.DataFrame):
        """EXACT same training logic as Colab."""
        logger.info("Starting model training...")
        
        # Preprocess data with exact logic
        X, y = self.preprocess_data_exact(training_df)
        
        if len(X) < 10:
            logger.error("Insufficient data for training (need at least 10 samples)")
            return None
        
        # Data size warnings - EXACT same logic
        if len(X) < 50:
            logger.warning(f"Small dataset ({len(X)} samples) - high risk of overfitting")
        elif len(X) < 100:
            logger.warning(f"Moderate dataset ({len(X)} samples) - use simple models")
        else:
            logger.info(f"Good dataset size ({len(X)} samples) for training")
        
        # Split data - EXACT same parameters
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features using RobustScaler - EXACT same
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection - EXACT same logic
        if len(X.columns) > 8:
            k_features = min(10, max(5, len(X.columns) // 3))
            self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            logger.info(f"Selected {len(selected_features)} best features (reduced from {len(X.columns)})")
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
            selected_features = X.columns.tolist()
            logger.info(f"Using all {len(selected_features)} features (no selection needed)")
        
        self.feature_names = selected_features
        
        # EXACT same models with same parameters
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=50, max_depth=5, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42)
        }
        
        # Add XGBoost if available
        if XGB_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=50, max_depth=4, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1
            )
        
        # Add LightGBM if available
        if LGB_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=50, max_depth=4, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbose=-1
            )
        
        best_model = None
        best_score = float('-inf')
        best_model_name = None
        best_cv_scores = None
        
        logger.info("Testing algorithms with cross-validation...")
        for name, model in models.items():
            try:
                # EXACT same cross-validation
                cv_scores = cross_val_score(
                    model, X_train_selected, y_train,
                    cv=min(5, len(X_train)), scoring='r2', n_jobs=-1
                )
                avg_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)
                logger.info(f"{name}: CV R² = {avg_cv_score:.4f} (±{std_cv_score:.4f})")
                
                # EXACT same selection criteria
                if avg_cv_score > best_score and std_cv_score < 0.3:
                    best_score = avg_cv_score
                    best_model = model
                    best_model_name = name
                    best_cv_scores = cv_scores
            except Exception as e:
                logger.error(f"{name} failed: {e}")
        
        if best_model is None:
            logger.warning("All models failed. Using basic Ridge regression.")
            best_model = Ridge(alpha=1.0, random_state=42)
            best_model_name = "RidgeRegression"
        
        # Train best model
        logger.info(f"Training best model: {best_model_name}")
        best_model.fit(X_train_selected, y_train)
        
        # Evaluate with exact same metrics
        y_pred_train = best_model.predict(X_train_selected)
        y_pred_test = best_model.predict(X_test_selected)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        logger.info(f"Training - R²: {train_r2:.4f}, MAE: {train_mae:.2f}")
        logger.info(f"Testing - R²: {test_r2:.4f}, MAE: {test_mae:.2f}")
        
        if best_cv_scores is not None:
            logger.info(f"Cross-Val - R²: {np.mean(best_cv_scores):.4f} (±{np.std(best_cv_scores):.4f})")
        
        # EXACT same overfitting detection
        overfitting_gap = train_r2 - test_r2
        if overfitting_gap > 0.2:
            logger.warning(f"Severe overfitting detected (gap: {overfitting_gap:.4f})")
        elif overfitting_gap > 0.1:
            logger.warning(f"Moderate overfitting detected (gap: {overfitting_gap:.4f})")
        else:
            logger.info(f"Good generalization (gap: {overfitting_gap:.4f})")
        
        # Save model
        self.model = best_model
        self.model_type = best_model_name
        
        model_data = {
            'model': self.model, 'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'model_type': self.model_type, 'feature_names': self.feature_names,
            'performance': {'test_r2': test_r2, 'test_mae': test_mae}
        }
        
        joblib.dump(model_data, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 Most Important Features:")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'model_type': best_model_name,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'cv_score': np.mean(best_cv_scores) if best_cv_scores is not None else None,
            'samples': len(X),
            'features': len(selected_features),
            'overfitting_gap': overfitting_gap,
            'train_r2': train_r2,
            'train_mae': train_mae
        }

# Global instances
scorer = None
r2_storage = None
turso_db = None
retraining_in_progress = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer, r2_storage, turso_db
    
    try:
        scorer = AdvancedRiderEfficiencyScorer()
        
        r2_storage = R2StorageManager(
            account_id=os.getenv("R2_ACCOUNT_ID"),
            access_key=os.getenv("R2_ACCESS_KEY"),
            secret_key=os.getenv("R2_SECRET_KEY"),
            bucket_name=os.getenv("R2_BUCKET_NAME", "ml-training-data")
        )
        
        turso_db = TursoManager(
            database_url=os.getenv("TURSO_DATABASE_URL"),
            auth_token=os.getenv("TURSO_AUTH_TOKEN")
        )
        
        # Create tables on startup
        await turso_db.create_tables()
        
        logger.info("ML service started successfully with Turso")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
    
    yield

app = FastAPI(title="ML Range Predictor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy" if scorer and scorer.model else "unhealthy",
        "model_loaded": scorer is not None and scorer.model is not None,
        "storage_connected": r2_storage is not None,
        "database_connected": turso_db is not None
    }

@app.post("/predict")
async def predict_and_store(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Main endpoint: predict + store + retrain with enhanced response."""
    try:
        file_content = await file.read()
        
        # 1. Get prediction
        prediction = await scorer.process_file_and_predict(file_content, file.filename)
        
        # 2. Store file
        storage_result = await r2_storage.upload_file(file_content, file.filename)
        
        # 3. Save analysis
        analysis_data = {**prediction, "filename": storage_result["filename"]}
        analysis_id = await turso_db.save_analysis(analysis_data)
        
        # 4. Queue retraining
        global retraining_in_progress
        if not retraining_in_progress:
            background_tasks.add_task(retrain_background)
        
        return {
            "predicted_range": prediction["predicted_range"],
            "efficiency_score": prediction["efficiency_score"],
            "analysis_id": analysis_id,
            "model_type": prediction["model_type"],
            "features_analyzed": prediction["features_analyzed"],
            "data_points": prediction["data_points"],
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_background():
    """R2-based retraining with EXACT Colab logic."""
    global retraining_in_progress, scorer
    
    try:
        retraining_in_progress = True
        logger.info("Starting R2-based retraining with exact Colab logic...")
        
        # 1. List all training files in R2
        training_files = await r2_storage.list_files()
        logger.info(f"Found {len(training_files)} files in R2")
        
        if len(training_files) < 2:
            logger.info("Not enough files for retraining (need at least 2)")
            return
        
        # 2. Download and load all files
        dataframes = []
        successful_downloads = 0
        
        for filename in training_files:
            try:
                logger.info(f"Downloading {filename}")
                file_content = await r2_storage.download_file(filename)
                
                # Read file based on extension
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(file_content))
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(io.BytesIO(file_content))
                else:
                    logger.warning(f"Skipping unsupported file type: {filename}")
                    continue
                
                dataframes.append(df)
                successful_downloads += 1
                
            except Exception as e:
                logger.error(f"Failed to download/read {filename}: {e}")
                continue
        
        logger.info(f"Successfully loaded {successful_downloads} files")
        
        if len(dataframes) < 2:
            logger.warning("Not enough valid dataframes for training")
            return
        
        # 3. Build training dataset with EXACT Colab logic
        training_df = scorer.build_training_dataset_from_r2(dataframes)
        
        if training_df.empty:
            logger.error("No valid training data generated")
            return
        
        logger.info(f"Training dataset: {len(training_df)} samples, {len(training_df.columns)-1} features")
        
        # 4. Train model with EXACT Colab logic
        training_stats = scorer.train_model_exact(training_df)
        
        if training_stats is None:
            logger.error("Training failed")
            return
        
        # 5. Reload model in memory
        scorer._load_model()
        logger.info("Reloaded new model in memory")
        
        # 6. Save training history to database
        try:
            await turso_db.execute_query("""
                INSERT INTO training_history 
                (model_type, samples_count, features_count, test_r2, test_mae, cv_score, 
                 files_processed, overfitting_gap, train_r2, train_mae, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, [
                training_stats['model_type'],
                training_stats['samples'],
                training_stats['features'],
                training_stats['test_r2'],
                training_stats['test_mae'],
                training_stats.get('cv_score'),
                successful_downloads,
                training_stats['overfitting_gap'],
                training_stats['train_r2'],
                training_stats['train_mae']
            ])
            
            logger.info("Saved training history to database")
        
        except Exception as db_error:
            logger.error(f"Failed to save training history: {db_error}")
        
        logger.info(f"Retraining completed! New {training_stats['model_type']} model trained on {successful_downloads} files")
        
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        retraining_in_progress = False

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


















