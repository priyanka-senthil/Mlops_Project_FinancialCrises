"""
Quick test after fixes - Tests only the critical path
"""
import os
import sys
import logging

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*80)
    print("🔧 QUICK FIX TEST - Testing VAE Loading")
    print("="*80)
    
    # Test Model Loading
    try:
        from config import Config
        from model_loader import GCSModelLoader
        from feature_mapper import FeatureMapper
        from gcs_data_fetcher import GCSDataFetcher
        from pipeline import StressTestPipeline
        
        print("\n✅ All imports successful")
        
        # Load models
        print("\n📥 Loading models...")
        loader = GCSModelLoader(
            bucket_name=Config.GCS_BUCKET,
            config=Config.MODEL_PATHS
        )
        models = loader.load_all_models()
        
        # Check Model 1
        print(f"\n📊 Model 1 Status:")
        print(f"   Features loaded: {models['model1']['n_features']}")
        print(f"   VAE present: {models['model1']['vae'] is not None}")
        print(f"   Scaler present: {models['model1']['scaler'] is not None}")
        
        if models['model1']['n_features'] == 0:
            print("\n❌ ERROR: VAE has 0 features!")
            print("   This means feature_names wasn't loaded correctly")
            return False
        
        # Load data
        print("\n📥 Loading data...")
        data_fetcher = GCSDataFetcher(
            bucket_name=Config.GCS_BUCKET,
            data_paths=Config.DATA_PATHS
        )
        data_fetcher.load_training_data()
        print(f"   ✓ {len(data_fetcher.company_lookup)} companies loaded")
        
        # Initialize pipeline
        print("\n🔧 Initializing pipeline...")
        mapper = FeatureMapper(Config.VAE_TO_MODEL2_MAPPING)
        pipeline = StressTestPipeline(
            models=models,
            feature_mapper=mapper,
            data_fetcher=data_fetcher,
            config=Config
        )
        print("   ✓ Pipeline initialized")
        
        # Test scenario generation
        print("\n🎲 Testing scenario generation...")
        try:
            scenarios = pipeline.generate_scenarios(n_scenarios=2)
            print(f"   ✅ Generated {len(scenarios)} scenarios!")
            
            for s in scenarios:
                print(f"\n   Scenario {s['scenario_id']}:")
                print(f"      Severity: {s['severity']}")
                print(f"      GDP: {s['features'].get('GDP', 0):.0f}")
                print(f"      VIX: {s['features'].get('VIX', 0):.2f}")
            
        except Exception as e:
            print(f"   ❌ Scenario generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test stress test
        print("\n🧪 Testing stress test...")
        try:
            company_id = list(data_fetcher.company_lookup.keys())[0]
            results = pipeline.run_stress_test(
                company_id=company_id,
                scenario_ids=[0]
            )
            
            if results:
                result = results[0]
                print(f"   ✅ Stress test successful!")
                print(f"   Risk Score: {result['risk_assessment']['risk_score']:.1f}")
                print(f"   Category: {result['risk_assessment']['risk_category']}")
            else:
                print(f"   ❌ No results returned")
                return False
                
        except Exception as e:
            print(f"   ❌ Stress test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("🚀 Ready to deploy")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)