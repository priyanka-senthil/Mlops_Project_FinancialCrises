
import functions_framework
from flask import jsonify, request
import os
import sys
import logging
import json

# CRITICAL: Set threading BEFORE any imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['LIGHTGBM_NUM_THREADS'] = '1'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your modules
try:
    from config import Config
    from model_loader import GCSModelLoader
    from feature_mapper import FeatureMapper
    from gcs_data_fetcher import GCSDataFetcher
    from pipeline import StressTestPipeline
    
    logger.info("✅ Successfully imported all modules")
except ImportError as e:
    logger.error(f"❌ Could not import modules: {e}")
    raise

# Global variables for caching (reused across invocations)
MODELS = None
PIPELINE = None
DATA_FETCHER = None
INITIALIZED = False


def initialize_pipeline():
    """Initialize the pipeline once and cache it"""
    global MODELS, PIPELINE, DATA_FETCHER, INITIALIZED
    
    if INITIALIZED:
        return True
    
    try:
        logger.info("🚀 Initializing Financial Stress Test API...")
        
        # Import torch and set single-threaded mode
        import torch
        torch.set_num_threads(1)
        logger.info(f"✅ PyTorch {torch.__version__} initialized (single-threaded)")
        
        # Load models from GCS
        logger.info("📥 Loading models from GCS...")
        model_loader = GCSModelLoader(
            bucket_name=Config.GCS_BUCKET,
            config=Config.MODEL_PATHS
        )
        MODELS = model_loader.load_all_models()
        logger.info("✅ Models loaded")
        
        # Load company data
        logger.info("📊 Loading company data...")
        DATA_FETCHER = GCSDataFetcher(
            bucket_name=Config.GCS_BUCKET,
            data_paths=Config.DATA_PATHS
        )
        DATA_FETCHER.load_training_data()
        logger.info("✅ Company data loaded")
        
        # Initialize pipeline
        logger.info("🔧 Initializing pipeline...")
        feature_mapper = FeatureMapper(Config.VAE_TO_MODEL2_MAPPING)
        
        PIPELINE = StressTestPipeline(
            models=MODELS,
            feature_mapper=feature_mapper,
            data_fetcher=DATA_FETCHER,
            config=Config
        )
        logger.info("✅ Pipeline initialized")
        
        # Pre-generate scenarios
        logger.info("🎲 Pre-generating scenarios...")
        PIPELINE.generate_scenarios(n_scenarios=Config.DEFAULT_N_SCENARIOS)
        logger.info(f"✅ Generated {len(PIPELINE.scenarios)} scenarios")
        
        INITIALIZED = True
        logger.info("✅ API READY")
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


@functions_framework.http
def predict(request):
    """
    Main entry point for stress test predictions
    
    Endpoints:
    - GET / or /health - Health check
    - GET /scenarios - List scenarios
    - POST /stress-test - Run stress test
    - POST /scenarios/generate - Generate new scenarios
    - GET /companies - List companies
    """
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
    }
    
    try:
        # Initialize on first request (cached after)
        if not INITIALIZED:
            logger.info("First request - initializing...")
            if not initialize_pipeline():
                return jsonify({
                    'error': 'Failed to initialize pipeline',
                    'status': 'failed'
                }), 500, headers
        
        # Route requests
        path = request.path
        method = request.method
        
        # Health check - default for root path
        if method == 'GET' and (path == '/' or path == '/health' or path == ''):
            return jsonify({
                'status': 'healthy',
                'service': 'Financial Stress Test API',
                'version': '1.0',
                'models_loaded': MODELS is not None,
                'pipeline_ready': PIPELINE is not None,
                'n_scenarios': len(PIPELINE.scenarios) if PIPELINE else 0,
                'n_companies': len(DATA_FETCHER.company_lookup) if DATA_FETCHER else 0,
                'endpoints': {
                    'health': 'GET /health',
                    'scenarios': 'GET /scenarios',
                    'companies': 'GET /companies',
                    'stress_test': 'POST /stress-test',
                    'generate': 'POST /scenarios/generate'
                }
            }), 200, headers
        
        # Get scenarios
        if method == 'GET' and 'scenarios' in path:
            scenarios_list = [
                {
                    "scenario_id": s["scenario_id"],
                    "severity": s["severity"],
                    "sigma": float(s["sigma"]),
                    "crisis_type": s.get("crisis_type", "Unknown"),
                    "preview": {
                        "GDP": float(s["features"].get("GDP", 0)),
                        "VIX": float(s["features"].get("VIX", 0)),
                        "Unemployment_Rate": float(s["features"].get("Unemployment_Rate", 0))
                    }
                }
                for s in PIPELINE.scenarios
            ]
            return jsonify({
                "scenarios": scenarios_list,
                "total": len(scenarios_list)
            }), 200, headers
        
        # Generate scenarios
        if method == 'POST' and 'generate' in path:
            request_json = request.get_json(silent=True)
            n_scenarios = request_json.get('n_scenarios', 10) if request_json else 10
            
            logger.info(f"📝 Generating {n_scenarios} scenarios")
            scenarios = PIPELINE.generate_scenarios(n_scenarios=n_scenarios)
            
            return jsonify({
                "message": f"Generated {len(scenarios)} scenarios",
                "n_scenarios": len(scenarios),
                "scenarios": [
                    {
                        "scenario_id": s["scenario_id"],
                        "severity": s["severity"],
                        "sigma": float(s["sigma"]),
                        "crisis_type": s.get("crisis_type", "Unknown")
                    }
                    for s in scenarios
                ]
            }), 200, headers
        
        # Run stress test
        if method == 'POST' and 'stress-test' in path:
            request_json = request.get_json(silent=True)
            
            if not request_json:
                return jsonify({
                    'error': 'No JSON data provided',
                    'example': {
                        'company_id': 'AAPL',
                        'scenario_ids': [0, 1, 2]
                    }
                }), 400, headers
            
            company_id = request_json.get('company_id')
            scenario_ids = request_json.get('scenario_ids', [])
            
            if not company_id:
                return jsonify({'error': 'company_id is required'}), 400, headers
            
            if company_id not in DATA_FETCHER.company_lookup:
                return jsonify({
                    'error': f'Company {company_id} not found'
                }), 404, headers
            
            # Run stress test
            logger.info(f"🧪 Running stress test for {company_id} on {len(scenario_ids)} scenarios")
            results = PIPELINE.run_stress_test(
                company_id=company_id,
                scenario_ids=scenario_ids
            )
            
            # Calculate aggregates if multiple scenarios
            if len(results) > 1:
                avg_risk = sum(r["risk_assessment"]["risk_score"] for r in results) / len(results)
                best_case = min(results, key=lambda x: x["risk_assessment"]["risk_score"])
                worst_case = max(results, key=lambda x: x["risk_assessment"]["risk_score"])
                
                return jsonify({
                    "company_id": company_id,
                    "n_scenarios": len(results),
                    "aggregated": True,
                    "summary": {
                        "avg_risk_score": round(avg_risk, 1),
                        "best_case": best_case,
                        "worst_case": worst_case
                    },
                    "detailed_results": results
                }), 200, headers
            else:
                return jsonify({
                    "company_id": company_id,
                    "n_scenarios": 1,
                    "aggregated": False,
                    "result": results[0]
                }), 200, headers
        
        # List companies
        if method == 'GET' and 'companies' in path:
            companies = [
                {
                    "company_id": cid,
                    "sector": data["sector"]
                }
                for cid, data in DATA_FETCHER.company_lookup.items()
            ]
            
            return jsonify({
                "companies": companies,
                "total": len(companies)
            }), 200, headers
        
        # Unknown endpoint
        return jsonify({
            'error': 'Unknown endpoint',
            'path': path,
            'method': method,
            'available_endpoints': [
                'GET / or /health - Health check',
                'GET /scenarios - List scenarios',
                'POST /stress-test - Run stress test',
                'POST /scenarios/generate - Generate scenarios',
                'GET /companies - List companies'
            ]
        }), 404, headers
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'failed',
            'type': type(e).__name__
        }), 500, headers