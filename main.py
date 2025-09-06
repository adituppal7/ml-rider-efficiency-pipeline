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
import asyncio
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import tempfile
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your existing AdvancedRiderEfficiencyScorer class with training capability
class AdvancedRiderEfficiencyScorer:
    """Advanced ML model with continuous learning capability."""

    def __init__(self, model_path: str = "efficiency_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.model_type = None
        self.feature_names = None
        self.training_history = []
        self.last_training_time = None
        
        if os.path.exists(self.model_path):
            self._load_model()

    def _load_model(self):
        """Load trained model and preprocessing components."""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data.get('feature_selector')
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.training_history = model_data.get('training_history', [])
            self.last_training_time = model_data.get('last_training_time')
            logger.info(f"Loaded {self.model_type} model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def extract_features(self, throttle_data: List[float],
                        current_data: Optional[List[float]] = None,
                        cell_temp_delta_data: Optional[List[float]] = None,
                        soc_data: Optional[List[float]] = None) -> Dict:
        """Extract features from ride data."""
        # Clean throttle data
        throttle = np.array([x for x in throttle_data if pd.notna(x) and 0 <= x <= 100])
        features = {}

        if len(throttle) == 0:
            return self._get_zero_features()

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

        # Efficiency-related features
        features['low_throttle_ratio'] = np.mean(throttle <= 20)
        features['moderate_throttle_ratio'] = np.mean((throttle > 20) & (throttle <= 60))
        features['high_throttle_ratio'] = np.mean((throttle > 60) & (throttle <= 80))
        features['aggressive_throttle_ratio'] = np.mean(throttle > 80)

        # Event counts
        features['eco_events'] = np.sum(throttle <= 20)
        features['moderate_events'] = np.sum((throttle > 20) & (throttle <= 60))
        features['high_events'] = np.sum((throttle > 60) & (throttle <= 80))
        features['aggressive_events'] = np.sum(throttle > 80)

        # Sustained behavior analysis
        features['max_sustained_eco'] = self._max_sustained_condition(throttle <= 20)
        features['max_sustained_moderate'] = self._max_sustained_condition((throttle > 20) & (throttle <= 60))
        features['max_sustained_high'] = self._max_sustained_condition(throttle > 60)
        features['max_sustained_aggressive'] = self._max_sustained_condition(throttle > 80)

        # Throttle change patterns
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

        # Current-based features
        if current_data is not None:
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

        # Temperature-based features
        if cell_temp_delta_data is not None:
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

        # SOC-based features
        if soc_data is not None:
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
        """Calculate maximum sustained duration of a condition."""
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

    def build_training_dataset_from_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Build training dataset from multiple files."""
        logger.info(f"Building dataset from {len(file_paths)} files...")
        
        rows = []
        for file_path in file_paths:
            try:
                # Read file
                if file_path.lower().endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                df.columns = df.columns.str.strip().str.lower()
                
                if 'throttle%' not in df.columns or 'range' not in df.columns:
                    logger.warning(f"Skipping {file_path}: Missing required columns")
                    continue
                
                throttle_data = pd.to_numeric(df['throttle%'], errors='coerce').dropna().tolist()
                
                current_data = None
                if 'current( a )' in df.columns:
                    current_data = pd.to_numeric(df['current( a )'], errors='coerce').dropna().tolist()
                
                temp_data = None
                if 'cell temp delta' in df.columns:
                    temp_data = pd.to_numeric(df['cell temp delta'], errors='coerce').dropna().tolist()
                
                soc_data = None
                soc_columns = ['soc( % )', 'soc(%)', 'soc %', 'soc%', 'soc']
                for col in soc_columns:
                    if col in df.columns:
                        soc_data = pd.to_numeric(df[col], errors='coerce').dropna().tolist()
                        break
                
                range_values = pd.to_numeric(df['range'], errors='coerce').dropna()
                if len(range_values) == 0 or len(throttle_data) < 10:
                    continue
                
                range_val = range_values.mean()
                if range_val <= 0 or range_val > 500:
                    continue
                
                features = self.extract_features(throttle_data, current_data, temp_data, soc_data)
                features['range'] = range_val
                features['source_file'] = os.path.basename(file_path)
                rows.append(features)
                
                logger.info(f"Processed {os.path.basename(file_path)}: range = {range_val:.1f} km")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if not rows:
            logger.error("No valid data found!")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(rows)
        logger.info(f"Built dataset: {len(result_df)} samples, {len(result_df.columns)-1} features")
        return result_df

    def retrain_model(self, training_df: pd.DataFrame) -> Dict[str, Any]:
        """Retrain model with new data."""
        logger.info("🚀 Starting model retraining...")
        
        if len(training_df) < 10:
            raise ValueError("Insufficient data for training (need at least 10 samples)")
        
        # Store training info
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'samples': len(training_df),
            'features': len(training_df.columns) - 1
        }
        
        # Your existing training logic here (simplified)
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.feature_selection import SelectKBest, f_regression
        import xgboost as xgb
        import lightgbm as lgb
        
        # Preprocess data
        feature_cols = [col for col in training_df.columns if col not in ['range', 'source_file']]
        X = training_df[feature_cols].fillna(0)
        y = training_df['range']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        if len(X.columns) > 8:
            k_features = min(10, max(5, len(X.columns) // 3))
            self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
            selected_features = X.columns.tolist()
            self.feature_selector = None
        
        self.feature_names = selected_features
        
        # Try multiple models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=50, max_depth=4, random_state=42, verbose=-1)
        }
        
        best_model = None
        best_score = float('-inf')
        best_model_name = None
        
        for name, model in models.items():
            try:
                cv_scores = cross_val_score(model, X_train_selected, y_train, cv=min(5, len(X_train)), scoring='r2')
                avg_cv_score = np.mean(cv_scores)
                
                if avg_cv_score > best_score:
                    best_score = avg_cv_score
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                logger.error(f"{name} failed: {e}")
        
        if best_model is None:
            from sklearn.linear_model import Ridge
            best_model = Ridge(alpha=1.0, random_state=42)
            best_model_name = "Ridge"
        
        # Train best model
        best_model.fit(X_train_selected, y_train)
        
        # Evaluate
        y_pred_test = best_model.predict(X_test_selected)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Update model
        self.model = best_model
        self.model_type = best_model_name
        self.last_training_time = datetime.now().isoformat()
        
        # Save training info
        training_info.update({
            'model_type': best_model_name,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'cv_score': best_score
        })
        self.training_history.append(training_info)
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'last_training_time': self.last_training_time
        }
        
        joblib.dump(model_data, self.model_path)
        logger.info(f"💾 Retrained model saved: {best_model_name} (R²: {test_r2:.4f})")
        
        return training_info

    def predict_range(self, throttle_data: List[float],
                     current_data: Optional[List[float]] = None,
                     cell_temp_delta_data: Optional[List[float]] = None,
                     soc_data: Optional[List[float]] = None) -> Dict[str, Any]:
        """Predict range from ride data."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # 1. Extract ALL features (41 features)
        features = self.extract_features(throttle_data, current_data, cell_temp_delta_data, soc_data)
        
        # 2. Convert to DataFrame with ALL features
        feature_df = pd.DataFrame([features])
        
        # 3. Reindex to match the scaler's expected features (from training)
        feature_df = feature_df.reindex(columns=self.scaler.feature_names_in_, fill_value=0)
        
        # 4. Scale ALL features
        features_scaled = self.scaler.transform(feature_df)
        
        # 5. Apply feature selection to get the 10 selected features
        if self.feature_selector is not None:
            features_selected = self.feature_selector.transform(features_scaled)
        else:
            features_selected = features_scaled
        
        # 6. Make prediction
        prediction = self.model.predict(features_selected)[0]
        confidence = min(100, max(0, 85))
        
        return {
            "predicted_range": round(float(prediction), 2),
            "confidence": round(confidence, 2),
            "model_type": self.model_type,
            "features_analyzed": len(features),
            "data_points": len(throttle_data),
            "last_training": self.last_training_time
        }

class GoogleDriveManager:
    """Manages Google Drive operations for training data."""
    
    def __init__(self, credentials_file: str, training_folder_id: str):
        self.credentials_file = credentials_file
        self.training_folder_id = training_folder_id
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Drive API."""
        try:
            if os.path.exists(self.credentials_file):
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_file,
                    scopes=['https://www.googleapis.com/auth/drive']
                )
                self.service = build('drive', 'v3', credentials=credentials)
                logger.info("✅ Google Drive authenticated successfully")
            else:
                logger.error(f"Credentials file not found: {self.credentials_file}")
        except Exception as e:
            logger.error(f"Google Drive authentication failed: {e}")
    
    def upload_file(self, file_path: str, filename: str) -> str:
        """Upload file to Google Drive training folder."""
        try:
            file_metadata = {
                'name': filename,
                'parents': [self.training_folder_id]
            }
            
            media = MediaFileUpload(file_path)
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            logger.info(f"✅ Uploaded {filename} to Google Drive")
            return file.get('id')
            
        except Exception as e:
            logger.error(f"Failed to upload {filename}: {e}")
            raise
    
    def download_all_training_files(self, local_dir: str) -> List[str]:
        """Download all files from training folder."""
        try:
            os.makedirs(local_dir, exist_ok=True)
            
            # List files in training folder
            results = self.service.files().list(
                q=f"'{self.training_folder_id}' in parents and trashed=false",
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            downloaded_files = []
            
            for file in files:
                file_id = file['id']
                filename = file['name']
                
                # Skip non-data files
                if not (filename.lower().endswith(('.csv', '.xlsx', '.xls'))):
                    continue
                
                local_path = os.path.join(local_dir, filename)
                
                # Download file
                request = self.service.files().get_media(fileId=file_id)
                with open(local_path, 'wb') as f:
                    downloader = MediaIoBaseDownload(f, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                
                downloaded_files.append(local_path)
                logger.info(f"📥 Downloaded {filename}")
            
            logger.info(f"✅ Downloaded {len(downloaded_files)} training files")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Failed to download training files: {e}")
            return []

# Pydantic models
class RideData(BaseModel):
    throttle_data: List[float]
    current_data: Optional[List[float]] = None
    cell_temp_delta_data: Optional[List[float]] = None
    soc_data: Optional[List[float]] = None

class RetrainingResponse(BaseModel):
    status: str
    message: str
    training_info: Optional[Dict[str, Any]] = None
    files_processed: int = 0

class PredictionResponse(BaseModel):
    predicted_range: float
    confidence: float
    model_type: str
    features_analyzed: int
    data_points: int
    last_training: Optional[str] = None
    status: str = "success"

# Global instances
scorer = None
drive_manager = None
retraining_in_progress = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer, drive_manager
    try:
        scorer = AdvancedRiderEfficiencyScorer()

        # Env vars
        credentials_json = os.getenv("GOOGLE_CREDENTIALS_FILE")
        training_folder_id = os.getenv("GOOGLE_DRIVE_TRAINING_FOLDER_ID")

        creds_path = None

        if credentials_json:
            creds_path = "google_credentials.json"
            with open(creds_path, "w") as f:
                f.write(credentials_json)
            logger.info("✅ Credentials JSON written to google_credentials.json")

        if creds_path and training_folder_id:
            drive_manager = GoogleDriveManager(creds_path, training_folder_id)
            logger.info("✅ Google Drive integration enabled")
        else:
            logger.warning("⚠️ Google Drive integration disabled - missing credentials or folder ID")

        logger.info("ML Service started successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")

    yield

    
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Continuous Learning ML Service",
    description="ML API with Google Drive integration and continuous learning",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check with training status."""
    global retraining_in_progress
    return {
        "status": "healthy" if scorer and scorer.model is not None else "unhealthy",
        "model_loaded": scorer is not None and scorer.model is not None,
        "model_type": scorer.model_type if scorer else None,
        "last_training": scorer.last_training_time if scorer else None,
        "training_in_progress": retraining_in_progress,
        "google_drive_enabled": drive_manager is not None,
        "training_history_count": len(scorer.training_history) if scorer else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_range(ride_data: RideData):
    """Predict vehicle range from ride data."""
    if scorer is None or scorer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = scorer.predict_range(
            throttle_data=ride_data.throttle_data,
            current_data=ride_data.current_data,
            cell_temp_delta_data=ride_data.cell_temp_delta_data,
            soc_data=ride_data.soc_data
        )
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-and-retrain")
async def upload_and_retrain(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload new training file and trigger retraining."""
    global retraining_in_progress
    
    if drive_manager is None:
        raise HTTPException(status_code=503, detail="Google Drive not configured")
    
    if retraining_in_progress:
        raise HTTPException(status_code=429, detail="Retraining already in progress")
    
    try:
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        # Upload to Google Drive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        drive_filename = f"{timestamp}_{file.filename}"
        drive_file_id = drive_manager.upload_file(temp_file.name, drive_filename)
        
        # Clean up temp file
        os.unlink(temp_file.name)
        
        # Trigger retraining in background
        background_tasks.add_task(retrain_model_background)
        retraining_in_progress = True
        
        return {
            "status": "success",
            "message": f"File {file.filename} uploaded to Google Drive as {drive_filename}",
            "drive_file_id": drive_file_id,
            "retraining_started": True
        }
        
    except Exception as e:
        logger.error(f"Upload and retrain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_model_background():
    """Background task for model retraining."""
    global retraining_in_progress, scorer
    
    try:
        logger.info("🚀 Starting background retraining...")
        
        # Download all training files
        temp_dir = tempfile.mkdtemp()
        training_files = drive_manager.download_all_training_files(temp_dir)
        
        if not training_files:
            logger.error("No training files found")
            return
        
        # Build training dataset
        training_df = scorer.build_training_dataset_from_files(training_files)
        
        if training_df.empty:
            logger.error("No valid training data found")
            return
        
        # Retrain model
        training_info = scorer.retrain_model(training_df)
        
        logger.info(f"✅ Retraining completed: {training_info}")
        
        # Clean up temp files
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        logger.error(f"Background retraining failed: {e}")
    finally:
        retraining_in_progress = False

@app.post("/manual-retrain", response_model=RetrainingResponse)
async def manual_retrain(background_tasks: BackgroundTasks):
    """Manually trigger model retraining with all Google Drive files."""
    global retraining_in_progress
    
    if drive_manager is None:
        raise HTTPException(status_code=503, detail="Google Drive not configured")
    
    if retraining_in_progress:
        raise HTTPException(status_code=429, detail="Retraining already in progress")
    
    # Trigger retraining in background
    background_tasks.add_task(retrain_model_background)
    retraining_in_progress = True
    
    return RetrainingResponse(
        status="started",
        message="Manual retraining started in background",
        files_processed=0
    )

@app.get("/training-history")
async def get_training_history():
    """Get model training history."""
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "training_history": scorer.training_history,
        "total_trainings": len(scorer.training_history),
        "last_training": scorer.last_training_time,
        "current_model": scorer.model_type
    }

@app.get("/training-status")
async def get_training_status():
    """Get current training status."""
    global retraining_in_progress
    
    return {
        "training_in_progress": retraining_in_progress,
        "model_loaded": scorer is not None and scorer.model is not None,
        "google_drive_enabled": drive_manager is not None,
        "last_training": scorer.last_training_time if scorer else None
    }

@app.post("/predict-file")
async def predict_from_file(file: UploadFile = File(...)):
    """Predict range from uploaded CSV/Excel file."""
    if scorer is None or scorer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read file
        content = await file.read()
        
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Extract data
        if 'throttle%' not in df.columns:
            raise HTTPException(status_code=400, detail="Missing 'throttle%' column")
        
        throttle_data = pd.to_numeric(df['throttle%'], errors='coerce').dropna().tolist()
        
        current_data = None
        if 'current( a )' in df.columns:
            current_data = pd.to_numeric(df['current( a )'], errors='coerce').dropna().tolist()
        
        temp_data = None
        if 'cell temp delta' in df.columns:
            temp_data = pd.to_numeric(df['cell temp delta'], errors='coerce').dropna().tolist()
        
        soc_data = None
        soc_columns = ['soc( % )', 'soc(%)', 'soc %', 'soc%', 'soc']
        for col in soc_columns:
            if col in df.columns:
                soc_data = pd.to_numeric(df[col], errors='coerce').dropna().tolist()
                break
        
        # Make prediction
        result = scorer.predict_range(throttle_data, current_data, temp_data, soc_data)
        
        return {
            "filename": file.filename,
            "prediction": result,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"File prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Continuous Learning ML API",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "Range prediction from ride data",
            "Google Drive integration",
            "Continuous learning",
            "Model retraining",
            "Training history tracking"
        ],
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_file": "/predict-file", 
            "upload_and_retrain": "/upload-and-retrain",
            "manual_retrain": "/manual-retrain",
            "training_history": "/training-history",
            "training_status": "/training-status",
            "docs": "/docs"
        }
    }

@app.post("/smart-process")
async def smart_process(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Get prediction AND trigger background retraining in one call."""
    global retraining_in_progress
    
    if scorer is None or scorer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 1. Get immediate prediction from current model
        content = await file.read()
        
        # Process file for prediction
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))

        # Clean column names - EXACT SAME LOGIC AS COLAB
        df.columns = df.columns.str.strip().str.lower()
        
        # Check required columns using SAME logic as training
        if 'throttle%' not in df.columns:
            raise HTTPException(status_code=400, detail="Missing 'throttle%' column")
        
        # Extract data using EXACT SAME column names as training
        throttle_data = pd.to_numeric(df['throttle%'], errors='coerce').dropna().tolist()
        
        current_data = None
        if 'current( a )' in df.columns:  # This will now match after .lower()
            current_data = pd.to_numeric(df['current( a )'], errors='coerce').dropna().tolist()
        
        temp_data = None
        if 'cell temp delta' in df.columns:
            temp_data = pd.to_numeric(df['cell temp delta'], errors='coerce').dropna().tolist()
        
        # SOC detection using SAME logic as Colab
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
        
        # Validate minimum data
        if len(throttle_data) < 5:
            raise HTTPException(status_code=400, detail="Insufficient throttle data points")
        
        # Get prediction using SAME feature extraction as training
        prediction_result = scorer.predict_range(throttle_data, current_data, temp_data, soc_data)
        
        # Background processing
        if drive_manager and not retraining_in_progress:
            background_tasks.add_task(upload_and_retrain_background, file.filename, content)
            logger.info(f"Started background processing for {file.filename}")
        
        return {
            "prediction": prediction_result,
            "file_processing": {
                "filename": file.filename,
                "status": "processed_and_saved", 
                "background_training": True,
                "message": "File saved to Google Drive, model improving in background"
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Smart process error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain-only")
async def retrain_only(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Retrain model with uploaded file (no Google Drive storage)."""
    global retraining_in_progress
    
    if retraining_in_progress:
        raise HTTPException(status_code=429, detail="Retraining already in progress")
    
    try:
        # Read the uploaded file
        content = await file.read()
        
        # Save to temporary file for processing
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_file.write(content)
        temp_file.close()
        
        # Start background retraining with just this file
        background_tasks.add_task(retrain_with_single_file, temp_file.name, file.filename)
        retraining_in_progress = True
        
        return {
            "status": "success",
            "message": f"Retraining started with {file.filename}",
            "note": "Model will be updated with this data only"
        }
        
    except Exception as e:
        logger.error(f"Retrain-only error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_with_single_file(file_path: str, filename: str):
    """Retrain model with single file."""
    global retraining_in_progress, scorer
    
    try:
        logger.info(f"Starting retraining with {filename}")
        
        # Build dataset from single file
        training_df = scorer.build_training_dataset_from_files([file_path])
        
        if training_df.empty:
            logger.error("No valid training data found")
            return
        
        # Retrain model
        training_info = scorer.retrain_model(training_df)
        
        # Clean up temp file
        os.unlink(file_path)
        
        logger.info(f"Retraining completed: {training_info}")
        
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
    finally:
        retraining_in_progress = False

@app.post("/bulk-retrain")
async def bulk_retrain(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Retrain model with multiple uploaded files at once."""
    global retraining_in_progress
    
    if retraining_in_progress:
        raise HTTPException(status_code=429, detail="Retraining already in progress")
    
    try:
        # Save all uploaded files temporarily
        temp_files = []
        filenames = []
        
        for file in files:
            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            
            temp_files.append(temp_file.name)
            filenames.append(file.filename)
        
        # Start retraining with all files
        background_tasks.add_task(retrain_with_multiple_files, temp_files, filenames)
        retraining_in_progress = True
        
        return {
            "status": "success",
            "message": f"Bulk retraining started with {len(files)} files",
            "files": filenames,
            "note": "Model will be trained on all uploaded files"
        }
        
    except Exception as e:
        logger.error(f"Bulk retrain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_with_multiple_files(file_paths: List[str], filenames: List[str]):
    """Retrain model with multiple files."""
    global retraining_in_progress, scorer
    
    try:
        logger.info(f"Starting bulk retraining with {len(file_paths)} files")
        
        # Build dataset from all files
        training_df = scorer.build_training_dataset_from_files(file_paths)
        
        if training_df.empty:
            logger.error("No valid training data found in uploaded files")
            return
        
        # Retrain model with all data
        training_info = scorer.retrain_model(training_df)
        
        # Clean up temp files
        for file_path in file_paths:
            try:
                os.unlink(file_path)
            except:
                pass
        
        logger.info(f"Bulk retraining completed: {training_info}")
        
    except Exception as e:
        logger.error(f"Bulk retraining failed: {e}")
    finally:
        retraining_in_progress = False


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)









