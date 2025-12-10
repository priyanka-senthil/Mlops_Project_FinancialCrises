"""
Load models from GCS bucket - FINAL FIXED VERSION
"""
import logging
import pickle
import json
import time
from pathlib import Path
from typing import Dict, Any
from google.cloud import storage
import io
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class VAE(nn.Module):
    """Variational Autoencoder - Must match training architecture"""
    def __init__(self, input_dim=72, latent_dim=32):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class GCSModelLoader:
    """Load models from GCS bucket"""
    
    def __init__(self, bucket_name: str, config: Dict):
        self.bucket_name = bucket_name
        self.config = config
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Local cache directory
        self.cache_dir = Path("/tmp/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, gcs_path: str, local_path: Path) -> Path:
        """Download single file from GCS"""
        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(str(local_path))
        logger.info(f"   ✓ Downloaded: {gcs_path}")
        return local_path
    
    def load_model1(self) -> Dict[str, Any]:
        """Load Model 1 (VAE Scenario Generator) - FINAL FIXED VERSION"""
        logger.info("📥 Loading Model 1: VAE Scenario Generator")
        
        config = self.config["model1"]
        gcs_base = config["gcs_path"]
        local_dir = self.cache_dir / "model1"
        local_dir.mkdir(exist_ok=True)
        
        # Download model file
        model_path = f"{gcs_base}{config['files']['model']}"
        local_model = local_dir / config['files']['model']
        self.download_file(model_path, local_model)
        
        # Load pickled dict
        with open(local_model, 'rb') as f:
            vae_dict = pickle.load(f)
        
        logger.info(f"   ℹ️  VAE dict keys: {list(vae_dict.keys())}")
        
        # Extract components - FIXED: Use correct keys
        scaler = vae_dict.get("scaler")
        
        # CRITICAL FIX: Try different possible keys for features
        features = None
        for key in ['feature_names', 'features', 'feature_list']:
            if key in vae_dict and vae_dict[key]:
                features = vae_dict[key]
                logger.info(f"   ✓ Found features under key '{key}'")
                break
        
        if not features:
            logger.error("   ❌ No features found in VAE dict!")
            # Use default feature list as fallback
            features = []
        
        vae_config = vae_dict.get("config", {})
        
        # CRITICAL FIX: Try different possible keys for model
        model_state = None
        for key in ['model', 'model_state_dict', 'state_dict']:
            if key in vae_dict and vae_dict[key]:
                model_state = vae_dict[key]
                logger.info(f"   ✓ Found model state under key '{key}'")
                break
        
        # Reconstruct VAE model
        input_dim = len(features) if features else 72  # Fallback to 72
        latent_dim = vae_config.get("latent_dim", 32)
        
        logger.info(f"   🔧 Reconstructing VAE: input={input_dim}, latent={latent_dim}")
        
        vae_model = VAE(input_dim=input_dim, latent_dim=latent_dim)
        
        if model_state:
            try:
                # Handle both state_dict and full model cases
                if isinstance(model_state, dict):
                    # It's a state_dict
                    vae_model.load_state_dict(model_state)
                    logger.info(f"   ✅ Loaded VAE from state_dict")
                elif hasattr(model_state, 'state_dict'):
                    # It's a full model
                    vae_model.load_state_dict(model_state.state_dict())
                    logger.info(f"   ✅ Loaded VAE from model object")
                else:
                    logger.warning(f"   ⚠️  Unknown model format: {type(model_state)}")
                
                vae_model.eval()
            except Exception as e:
                logger.warning(f"   ⚠️  Failed to load model state: {e}")
                logger.info(f"   ℹ️  Using fresh VAE model")
        else:
            logger.warning(f"   ⚠️  No model state found, using fresh model")
        
        logger.info(f"   ✅ VAE ready with {len(features)} features")
        
        return {
            "vae": vae_model,
            "scaler": scaler,
            "features": features,
            "config": vae_config,
            "type": "vae",
            "n_features": len(features)
        }
    
    def load_model2(self) -> Dict[str, Any]:
        """Load Model 2 (Predictive Models - 5 targets)"""
        logger.info("📥 Loading Model 2: Predictive Models (5 targets)")
        
        config = self.config["model2"]
        gcs_base = config["gcs_path"]
        local_dir = self.cache_dir / "model2"
        local_dir.mkdir(exist_ok=True)
        
        models = {}
        scalers = {}
        feature_names = None
        
        # Load each target's best model
        for target, filename in config["targets"].items():
            gcs_path = f"{gcs_base}{filename}"
            local_path = local_dir / filename
            
            self.download_file(gcs_path, local_path)
            
            with open(local_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check structure
            if isinstance(model_data, dict):
                # Try to find model in dict
                model_obj = None
                for key in ['model', 'best_model', 'estimator', 'regressor']:
                    if key in model_data:
                        model_obj = model_data[key]
                        break
                
                if model_obj:
                    models[target] = model_obj
                    scalers[target] = model_data.get('scaler')
                    
                    # Get feature names (try different keys)
                    if feature_names is None:
                        for key in ['feature_names', 'features', 'feature_list']:
                            if key in model_data and model_data[key]:
                                feature_names = model_data[key]
                                logger.info(f"   ✓ Found {len(feature_names)} feature names from '{key}'")
                                break
                    
                    logger.info(f"   ✓ {target:15s} loaded")
                else:
                    logger.error(f"   ❌ {target}: No model found in keys: {list(model_data.keys())}")
                    models[target] = None
            else:
                # Direct model object
                models[target] = model_data
                scalers[target] = None
                logger.info(f"   ✓ {target:15s} loaded (direct)")
        
        return {
            "models": models,
            "scalers": scalers,
            "feature_names": feature_names,
            "type": config["type"],
            "n_features": config["n_features"]
        }
    
    def load_model3(self) -> Dict[str, Any]:
        """Load Model 3 (Anomaly Detection)"""
        logger.info("📥 Loading Model 3: Anomaly Detection")
        
        config = self.config["model3"]
        gcs_base = config["gcs_path"]
        local_dir = self.cache_dir / "model3"
        local_dir.mkdir(exist_ok=True)
        
        # Download model
        model_path = f"{gcs_base}{config['files']['model']}"
        local_model = local_dir / config['files']['model']
        self.download_file(model_path, local_model)
        
        with open(local_model, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler = None
        scaler_path = f"{gcs_base}{config['files']['scaler']}"
        local_scaler = local_dir / config['files']['scaler']
        try:
            self.download_file(scaler_path, local_scaler)
            with open(local_scaler, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"   ✓ Loaded scaler")
        except Exception as e:
            logger.warning(f"   ⚠️  No scaler found: {e}")
        
        # Load features
        features_path = f"{gcs_base}{config['files']['features']}"
        local_features = local_dir / config['files']['features']
        self.download_file(features_path, local_features)
        
        with open(local_features, 'r') as f:
            features_data = json.load(f)
            features = features_data.get("features", [])
        
        logger.info(f"   ✓ Loaded model ({len(features)} features)")
        
        return {
            "model": model,
            "scaler": scaler,
            "features": features,
            "type": config["type"],
            "n_features": len(features)
        }
    
    def load_all_models(self) -> Dict[str, Any]:
        """Load all three models SEQUENTIALLY with delays"""
        logger.info("="*80)
        logger.info("🚀 LOADING ALL MODELS FROM GCS")
        logger.info("="*80)
        
        try:
            # Set PyTorch to single thread
            torch.set_num_threads(1)
            
            # Load Model 1 (VAE)
            logger.info("\n⏳ Loading Model 1...")
            model1 = self.load_model1()
            time.sleep(1)
            logger.info("✅ Model 1 loaded and stabilized")
            
            # Load Model 2 (Predictive)
            logger.info("\n⏳ Loading Model 2...")
            model2 = self.load_model2()
            time.sleep(1)
            logger.info("✅ Model 2 loaded")
            
            # Load Model 3 (Anomaly)
            logger.info("\n⏳ Loading Model 3...")
            model3 = self.load_model3()
            logger.info("✅ Model 3 loaded")
            
            logger.info("\n" + "="*80)
            logger.info("✅ ALL MODELS LOADED SUCCESSFULLY")
            logger.info("="*80)
            
            return {
                "model1": model1,
                "model2": model2,
                "model3": model3
            }
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise