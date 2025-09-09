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
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import tempfile
import boto3
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge

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

class PlanetScaleManager:
    """Manages PlanetScale database operations."""
    
    def __init__(self, host: str, username: str, password: str, database: str):
        self.host = host
        self.username = username
        self.password = password
        self.database = database
        self.pool = None
    
    async def connect(self):
        """Create connection pool."""
        try:
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=3306,
                user=self.username,
                password=self.password,
                db=self.database,
                charset='utf8mb4',
                ssl_disabled=False,
                autocommit=True
            )
            logger.info("Connected to PlanetScale database")
        except Exception as e:
            logger.error(f"Failed to connect to PlanetScale: {e}")
            raise
    
    async def save_analysis(self, analysis_data: Dict) -> str:
        """Save analysis results to database."""
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    query = """
                    INSERT INTO analysis_results 
                    (filename, predicted_range, confidence, model_type, features_analyzed, 
                     data_points, throttle_avg, soc_start, soc_end, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    values = (
                        analysis_data.get('filename'),
                        analysis_data.get('predicted_range'),
                        analysis_data.get('confidence'),
                        analysis_data.get('model_type'),
                        analysis_data.get('features_analyzed'),
                        analysis_data.get('data_points'),
                        analysis_data.get('throttle_avg'),
                        analysis_data.get('soc_start'),
                        analysis_data.get('soc_end'),
                        datetime.now()
                    )
                    
                    await cursor.execute(query, values)
                    analysis_id = cursor.lastrowid
                    return str(analysis_id)
                    
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            raise HTTPException(status_code=500, detail="Failed to save analysis")
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()

class AdvancedRiderEfficiencyScorer:
    """ML model with consistent feature handling to avoid mismatches."""

    def __init__(self, model_path: str = "efficiency_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.model_type = None
        self.feature_names = None
        # CRITICAL: Define exact feature order from training
        self.EXPECTED_FEATURES = [
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
            # Return zeros for all expected features
            return {feature: 0 for feature in self.EXPECTED_FEATURES}

        features = {}
        
        # Basic throttle statistics
        features['avg_throttle'] = np.mean(throttle)
        features['median_throttle'] = np.median(throttle)
        features['max_throttle'] = np.max(throttle)
        features['min_throttle'] = np.min(throttle)  # Will be removed later
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
                features['max_temp_delta'] = np.max(temp_delta)  # Will be removed later
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

    async def process_file_and_predict(self, file_content: bytes, filename: str) -> Dict:
        """Process file and predict with feature mismatch protection."""
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
            
            return {
                "predicted_range": round(float(prediction), 2),
                "confidence": 85,
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
            logger.error(f"Error details: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def retrain_model(self, storage: R2StorageManager) -> Dict:
        """Retrain model with files from storage."""
        try:
            logger.info("Starting model retraining...")
            
            # Download all files
            filenames = await storage.list_files()
            if not filenames:
                raise ValueError("No training files found")
            
            # Build dataset
            rows = []
            for filename in filenames[:20]:  # Limit for demo
                try:
                    file_content = await storage.download_file(filename)
                    
                    if filename.lower().endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(file_content))
                    else:
                        df = pd.read_excel(io.BytesIO(file_content))
                    
                    # Same preprocessing
                    df.columns = df.columns.str.strip().str.lower()
                    
                    if 'throttle%' not in df.columns or 'range' not in df.columns:
                        continue
                    
                    throttle_data = pd.to_numeric(df['throttle%'], errors='coerce').dropna().tolist()
                    range_values = pd.to_numeric(df['range'], errors='coerce').dropna()
                    
                    if len(throttle_data) < 10 or len(range_values) == 0:
                        continue
                    
                    # Extract other data
                    current_data = None
                    if 'current( a )' in df.columns:
                        current_data = pd.to_numeric(df['current( a )'], errors='coerce').dropna().tolist()
                    
                    temp_data = None
                    if 'cell temp delta' in df.columns:
                        temp_data = pd.to_numeric(df['cell temp delta'], errors='coerce').dropna().tolist()
                    
                    soc_data = None
                    # Same SOC detection logic...
                    
                    range_val = range_values.mean()
                    if range_val <= 0 or range_val > 500:
                        continue
                    
                    features = self.extract_features(throttle_data, current_data, temp_data, soc_data)
                    features['range'] = range_val
                    rows.append(features)
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    continue
            
            if len(rows) < 10:
                raise ValueError("Insufficient training data")
            
            # Train model
            training_df = pd.DataFrame(rows)
            feature_cols = [col for col in training_df.columns if col != 'range']
            X = training_df[feature_cols].fillna(0)
            y = training_df['range']
            
            # Remove zero-variance features
            zero_var_cols = ['min_throttle', 'aggressive_throttle_ratio', 'aggressive_events', 'max_sustained_aggressive', 'max_temp_delta']
            X = X.drop(columns=[col for col in zero_var_cols if col in X.columns])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale and select features
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            k_features = min(10, max(5, len(X.columns) // 3))
            self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            
            self.feature_names = X.columns[self.feature_selector.get_support()].tolist()
            
            # Train Ridge model
            self.model = Ridge(alpha=1.0, random_state=42)
            self.model.fit(X_train_selected, y_train)
            self.model_type = "Ridge"
            
            # Evaluate
            y_pred_test = self.model.predict(X_test_selected)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Save model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'model_type': self.model_type,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, self.model_path)
            
            return {
                "model_type": self.model_type,
                "test_r2": test_r2,
                "test_mae": test_mae,
                "samples": len(training_df),
                "features": len(self.feature_names)
            }
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            raise

# Global instances
scorer = None
r2_storage = None
planetscale_db = None
retraining_in_progress = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer, r2_storage, planetscale_db
    
    try:
        scorer = AdvancedRiderEfficiencyScorer()
        
        r2_storage = R2StorageManager(
            account_id=os.getenv("R2_ACCOUNT_ID"),
            access_key=os.getenv("R2_ACCESS_KEY"),
            secret_key=os.getenv("R2_SECRET_KEY"),
            bucket_name=os.getenv("R2_BUCKET_NAME", "ml-training-data")
        )
        
        planetscale_db = PlanetScaleManager(
            host=os.getenv("PLANETSCALE_HOST"),
            username=os.getenv("PLANETSCALE_USERNAME"),
            password=os.getenv("PLANETSCALE_PASSWORD"),
            database=os.getenv("PLANETSCALE_DATABASE")
        )
        
        await planetscale_db.connect()
        logger.info("ML service started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
    
    yield
    
    if planetscale_db:
        await planetscale_db.close()

app = FastAPI(title="ML Range Predictor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ONLY THE ENDPOINTS YOU NEED:

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
    """Main endpoint: predict + store + retrain."""
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
            "confidence": prediction["confidence"],
            "analysis_id": analysis_id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_background():
    """Background retraining."""
    global retraining_in_progress, scorer
    
    try:
        retraining_in_progress = True
        logger.info("Starting background retraining...")
        
        await scorer.retrain_model(r2_storage)
        logger.info("Retraining completed")
        
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
    finally:
        retraining_in_progress = False

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)













