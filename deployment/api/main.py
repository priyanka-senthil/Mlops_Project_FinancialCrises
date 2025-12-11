# """
# ============================================================================
# Financial Stress Test Platform - Main API Entry Point
# ============================================================================
# Integrates YOUR stress test pipeline with enterprise features
# Author: Parth Saraykar
# Version: 3.0.0 - Production Ready
# ============================================================================
# """

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from datetime import datetime
# import sys
# from pathlib import Path

# # Add parent directory to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent))

# # ============================================================================
# # IMPORT ROUTERS
# # ============================================================================

# # Try to import auth system
# try:
#     from auth_system import router as auth_router
#     AUTH_LOADED = True
# except ImportError as e:
#     print(f"⚠️  Warning: Could not load auth_system.py: {e}")
#     print("    Running without authentication")
#     AUTH_LOADED = False

# # Try to import production features
# try:
#     from production_features import router as prod_router
#     PRODUCTION_LOADED = True
# except ImportError as e:
#     print(f"⚠️  Warning: Could not load production_features.py: {e}")
#     print("    Running without production features")
#     PRODUCTION_LOADED = False

# # ============================================================================
# # CREATE FASTAPI APPLICATION
# # ============================================================================

# app = FastAPI(
#     title="Financial Stress Test Platform",
#     description="Enterprise Risk Analytics with ML-Powered Stress Testing",
#     version="3.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc",
#     contact={
#         "name": "Parth Saraykar",
#         "email": "parth@financialstress.com"
#     }
# )

# # ============================================================================
# # CONFIGURE CORS
# # ============================================================================

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production: specify exact origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ============================================================================
# # INCLUDE ROUTERS
# # ============================================================================

# if AUTH_LOADED:
#     app.include_router(auth_router)

# if PRODUCTION_LOADED:
#     app.include_router(prod_router)

# # ============================================================================
# # ROOT ENDPOINTS
# # ============================================================================

# @app.get("/")
# async def root():
#     """Root endpoint - API information"""
#     return {
#         "message": "Financial Stress Test Platform API",
#         "version": "3.0.0",
#         "status": "operational",
#         "timestamp": datetime.now().isoformat(),
#         "documentation": {
#             "swagger_ui": "/docs",
#             "redoc": "/redoc"
#         },
#         "features": {
#             "authentication": AUTH_LOADED,
#             "stress_testing": PRODUCTION_LOADED,
#             "batch_processing": PRODUCTION_LOADED,
#             "risk_limits": PRODUCTION_LOADED,
#             "reports": PRODUCTION_LOADED,
#             "portfolio_analysis": PRODUCTION_LOADED
#         },
#         "endpoints": {
#             "health": "/api/v1/health",
#             "login": "/api/v1/auth/login",
#             "stress_test": "/api/v1/stress-test",
#             "scenarios": "/api/v1/scenarios"
#         }
#     }

# @app.get("/health")
# async def health_check():
#     """Simple health check"""
#     return {
#         "status": "healthy",
#         "service": "Financial Stress Test Platform",
#         "version": "3.0.0",
#         "timestamp": datetime.now().isoformat()
#     }

# # ============================================================================
# # ERROR HANDLERS
# # ============================================================================

# @app.exception_handler(404)
# async def not_found_handler(request, exc):
#     """Custom 404 error handler"""
#     return {
#         "error": "Not Found",
#         "message": f"The endpoint {request.url.path} does not exist",
#         "documentation": "/docs",
#         "status_code": 404
#     }

# @app.exception_handler(500)
# async def internal_error_handler(request, exc):
#     """Custom 500 error handler"""
#     return {
#         "error": "Internal Server Error",
#         "message": "An unexpected error occurred. Check server logs.",
#         "status_code": 500
#     }

# # ============================================================================
# # STARTUP EVENT
# # ============================================================================

# @app.on_event("startup")
# async def startup_event():
#     """Execute on API startup"""
#     print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
#     print("🚀 Financial Stress Test Platform API")
#     print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
#     print(f"📊 Version: 3.0.0")
#     print(f"🌐 API Documentation: http://localhost:8000/docs")
#     print(f"📖 Alternative Docs: http://localhost:8000/redoc")
#     print(f"🔐 Authentication: {'Enabled' if AUTH_LOADED else 'Disabled'}")
#     print(f"✅ Production Features: {'Loaded' if PRODUCTION_LOADED else 'Not Loaded'}")
#     print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
#     # Check GCS connectivity
#     try:
#         from google.cloud import storage
#         client = storage.Client()
#         bucket = client.bucket("mlops-financial-stress-data")
        
#         if bucket.exists():
#             print("✅ GCS Bucket: Connected")
#         else:
#             print("⚠️  GCS Bucket: Not found")
#     except Exception as e:
#         error_msg = str(e)[:60]
#         print(f"⚠️  GCS: {error_msg}...")
#         print("    (This is OK for local testing)")
    
#     print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
#     print("✨ Server ready! Press CTRL+C to quit")
#     print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Execute on API shutdown"""
#     print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
#     print("🛑 Shutting down Financial Stress Test Platform API")
#     print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# # ============================================================================
# # RUN SERVER
# # ============================================================================

# if __name__ == "__main__":
#     import uvicorn
    
#     uvicorn.run(
#         app, 
#         host="0.0.0.0", 
#         port=8000,
#         log_level="info"
#     )



# """
# ============================================================================
# Financial Stress Test Platform - Main API Entry Point
# ============================================================================
# Integrates YOUR stress test pipeline with enterprise features
# Author: Parth Saraykar
# Version: 3.0.1 - Cloud Run Production Ready
# ============================================================================
# """

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from contextlib import asynccontextmanager
# from datetime import datetime
# import sys
# import os
# from pathlib import Path

# # Add parent directory to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent))

# # ============================================================================
# # IMPORT ROUTERS
# # ============================================================================

# # Try to import auth system
# try:
#     from auth_system import router as auth_router
#     AUTH_LOADED = True
# except ImportError as e:
#     print(f"⚠️  Warning: Could not load auth_system.py: {e}")
#     print("    Running without authentication")
#     AUTH_LOADED = False

# # Try to import production features
# try:
#     from production_features import router as prod_router
#     PRODUCTION_LOADED = True
# except ImportError as e:
#     print(f"⚠️  Warning: Could not load production_features.py: {e}")
#     print("    Running without production features")
#     PRODUCTION_LOADED = False

# # ============================================================================
# # LIFESPAN CONTEXT MANAGER (Modern FastAPI)
# # ============================================================================

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Startup and shutdown events"""
#     # Startup
#     print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
#     print("🚀 Financial Stress Test Platform API")
#     print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
#     print(f"📊 Version: 3.0.1 (Cloud Run)")
#     print(f"🔐 Authentication: {'Enabled' if AUTH_LOADED else 'Disabled'}")
#     print(f"✅ Production Features: {'Loaded' if PRODUCTION_LOADED else 'Not Loaded'}")
    
#     # Check GCS connectivity
#     try:
#         from google.cloud import storage
#         client = storage.Client()
#         bucket = client.bucket("mlops-financial-stress-data")
        
#         if bucket.exists():
#             print("✅ GCS Bucket: Connected")
#         else:
#             print("⚠️  GCS Bucket: Not found")
#     except Exception as e:
#         error_msg = str(e)[:60]
#         print(f"⚠️  GCS: {error_msg}...")
#         print("    (This is OK for local testing)")
    
#     print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
#     print("✨ Server ready!")
#     print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
#     yield
    
#     # Shutdown
#     print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
#     print("🛑 Shutting down Financial Stress Test Platform API")
#     print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# # ============================================================================
# # CREATE FASTAPI APPLICATION
# # ============================================================================

# app = FastAPI(
#     title="Financial Stress Test Platform",
#     description="Enterprise Risk Analytics with ML-Powered Stress Testing",
#     version="3.0.1",
#     docs_url="/docs",
#     redoc_url="/redoc",
#     lifespan=lifespan,
#     contact={
#         "name": "Parth Saraykar",
#         "email": "parth@financialstress.com"
#     }
# )

# # ============================================================================
# # CONFIGURE CORS
# # ============================================================================

# # Get allowed origins from environment variable
# ALLOWED_ORIGINS = os.getenv(
#     "ALLOWED_ORIGINS",
#     "https://storage.googleapis.com,https://mlops-financial-stress-ui.storage.googleapis.com"
# ).split(",")

# # For local development, add localhost
# if os.getenv("ENVIRONMENT", "production") == "development":
#     ALLOWED_ORIGINS.extend([
#         "http://localhost:3000",
#         "http://localhost:8000",
#         "http://127.0.0.1:3000",
#         "http://127.0.0.1:8000"
#     ])

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=ALLOWED_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ============================================================================
# # INCLUDE ROUTERS
# # ============================================================================

# if AUTH_LOADED:
#     app.include_router(auth_router)

# if PRODUCTION_LOADED:
#     app.include_router(prod_router)

# # ============================================================================
# # ROOT ENDPOINTS
# # ============================================================================

# @app.get("/")
# async def root():
#     """Root endpoint - API information"""
#     return {
#         "message": "Financial Stress Test Platform API",
#         "version": "3.0.1",
#         "status": "operational",
#         "timestamp": datetime.now().isoformat(),
#         "environment": os.getenv("ENVIRONMENT", "production"),
#         "documentation": {
#             "swagger_ui": "/docs",
#             "redoc": "/redoc"
#         },
#         "features": {
#             "authentication": AUTH_LOADED,
#             "stress_testing": PRODUCTION_LOADED,
#             "batch_processing": PRODUCTION_LOADED,
#             "risk_limits": PRODUCTION_LOADED,
#             "reports": PRODUCTION_LOADED,
#             "portfolio_analysis": PRODUCTION_LOADED
#         },
#         "endpoints": {
#             "health": "/health",
#             "api_health": "/api/v1/health",
#             "login": "/api/v1/auth/login",
#             "stress_test": "/api/v1/stress-test",
#             "scenarios": "/api/v1/scenarios"
#         }
#     }

# @app.get("/health")
# async def health_check():
#     """Simple health check for load balancers"""
#     return {
#         "status": "healthy",
#         "service": "Financial Stress Test Platform",
#         "version": "3.0.1",
#         "timestamp": datetime.now().isoformat()
#     }

# @app.get("/api/v1/health")
# async def detailed_health():
#     """Detailed health check with component status"""
#     health_status = {
#         "status": "healthy",
#         "service": "Financial Stress Test Platform",
#         "version": "3.0.1",
#         "timestamp": datetime.now().isoformat(),
#         "components": {
#             "api": "healthy",
#             "auth": "loaded" if AUTH_LOADED else "not_loaded",
#             "production_features": "loaded" if PRODUCTION_LOADED else "not_loaded"
#         }
#     }
    
#     # Check GCS connectivity
#     try:
#         from google.cloud import storage
#         client = storage.Client()
#         bucket = client.bucket("mlops-financial-stress-data")
#         bucket.exists()
#         health_status["components"]["gcs"] = "healthy"
#     except Exception as e:
#         health_status["components"]["gcs"] = "degraded"
#         health_status["status"] = "degraded"
    
#     return health_status

# # ============================================================================
# # ERROR HANDLERS
# # ============================================================================

# @app.exception_handler(404)
# async def not_found_handler(request, exc):
#     """Custom 404 error handler"""
#     return {
#         "error": "Not Found",
#         "message": f"The endpoint {request.url.path} does not exist",
#         "documentation": "/docs",
#         "status_code": 404
#     }

# @app.exception_handler(500)
# async def internal_error_handler(request, exc):
#     """Custom 500 error handler"""
#     return {
#         "error": "Internal Server Error",
#         "message": "An unexpected error occurred. Check server logs.",
#         "status_code": 500
#     }

# # ============================================================================
# # RUN SERVER (LOCAL DEVELOPMENT)
# # ============================================================================

# if __name__ == "__main__":
#     import uvicorn
    
#     # Cloud Run provides PORT environment variable
#     port = int(os.getenv("PORT", 8000))
    
#     uvicorn.run(
#         app, 
#         host="0.0.0.0", 
#         port=port,
#         log_level="info"
#     )


"""
============================================================================
Financial Stress Test Platform - Main API Entry Point
============================================================================
Integrates YOUR stress test pipeline with enterprise features
Author: Parth Saraykar
Version: 3.0.1 - Cloud Run Production Ready
============================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import sys
import os
from pathlib import Path
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# IMPORT ROUTERS WITH DETAILED ERROR REPORTING
# ============================================================================

print("=" * 80)
print("🔍 LOADING AUTHENTICATION SYSTEM...")
print("=" * 80)

AUTH_LOADED = False
auth_router = None

try:
    print("   → Step 1: Importing auth_system module...")
    from auth_system import router as auth_router
    print("   ✅ Module imported successfully!")
    
    print("   → Step 2: Verifying router object...")
    if auth_router is None:
        raise Exception("Router is None after import")
    print("   ✅ Router object is valid!")
    
    print("   → Step 3: Checking router type...")
    print(f"      Router type: {type(auth_router)}")
    print("   ✅ Router type confirmed!")
    
    AUTH_LOADED = True
    print("\n   ✅✅✅ AUTH SYSTEM LOADED SUCCESSFULLY! ✅✅✅\n")
    
except ImportError as e:
    print(f"\n   ❌ ImportError: {e}")
    print("\n   Full traceback:")
    print(traceback.format_exc())
    print("\n   ⚠️  AUTH WILL BE DISABLED\n")
    AUTH_LOADED = False
    
except Exception as e:
    print(f"\n   ❌ Unexpected error: {e}")
    print("\n   Full traceback:")
    print(traceback.format_exc())
    print("\n   ⚠️  AUTH WILL BE DISABLED\n")
    AUTH_LOADED = False

print("=" * 80)
print(f"🔐 FINAL AUTH STATUS: {'✅ ENABLED' if AUTH_LOADED else '❌ DISABLED'}")
print("=" * 80)

# Try to import production features
print("\n" + "=" * 80)
print("🔍 LOADING PRODUCTION FEATURES...")
print("=" * 80)

PRODUCTION_LOADED = False
prod_router = None

try:
    print("   → Importing production_features module...")
    from production_features import router as prod_router
    print("   ✅ Production features imported!")
    PRODUCTION_LOADED = True
    
except ImportError as e:
    print(f"   ⚠️  Warning: Could not load production_features.py: {e}")
    print("   Running without production features")
    PRODUCTION_LOADED = False
    
except Exception as e:
    print(f"   ⚠️  Error loading production features: {e}")
    PRODUCTION_LOADED = False

print("=" * 80)
print(f"📦 PRODUCTION FEATURES: {'✅ LOADED' if PRODUCTION_LOADED else '❌ NOT LOADED'}")
print("=" * 80)
print("\n")

# ============================================================================
# LIFESPAN CONTEXT MANAGER (Modern FastAPI)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🚀 Financial Stress Test Platform API")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📊 Version: 3.0.1 (Cloud Run)")
    print(f"🔐 Authentication: {'✅ ENABLED' if AUTH_LOADED else '❌ DISABLED'}")
    print(f"📦 Production Features: {'✅ LOADED' if PRODUCTION_LOADED else '❌ NOT LOADED'}")
    
    # Check GCS connectivity
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket("mlops-financial-stress-data")
        
        if bucket.exists():
            print("✅ GCS Bucket: Connected")
        else:
            print("⚠️  GCS Bucket: Not found")
    except Exception as e:
        error_msg = str(e)[:60]
        print(f"⚠️  GCS: {error_msg}...")
        print("    (This is OK for local testing)")
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✨ Server ready!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    yield
    
    # Shutdown
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🛑 Shutting down Financial Stress Test Platform API")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# ============================================================================
# CREATE FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Financial Stress Test Platform",
    description="Enterprise Risk Analytics with ML-Powered Stress Testing",
    version="3.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "Parth Saraykar",
        "email": "parth@financialstress.com"
    }
)

# ============================================================================
# CONFIGURE CORS
# ============================================================================

# Get allowed origins from environment variable
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://storage.googleapis.com,https://mlops-financial-stress-ui.storage.googleapis.com,https://financial-stress-login.storage.googleapis.com"
).split(",")

# For local development, add localhost
if os.getenv("ENVIRONMENT", "production") == "development":
    ALLOWED_ORIGINS.extend([
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# INCLUDE ROUTERS
# ============================================================================

print("=" * 80)
print("🔧 REGISTERING ROUTERS WITH FASTAPI APP...")
print("=" * 80)

if AUTH_LOADED and auth_router is not None:
    try:
        print("   → Including auth router...")
        app.include_router(auth_router)
        print("   ✅ Auth router registered successfully!")
    except Exception as e:
        print(f"   ❌ Failed to register auth router: {e}")
        print(traceback.format_exc())
else:
    print("   ⚠️  Skipping auth router (not loaded)")

if PRODUCTION_LOADED and prod_router is not None:
    try:
        print("   → Including production features router...")
        app.include_router(prod_router)
        print("   ✅ Production router registered successfully!")
    except Exception as e:
        print(f"   ❌ Failed to register production router: {e}")
        print(traceback.format_exc())
else:
    print("   ⚠️  Skipping production router (not loaded)")

print("=" * 80)
print(f"📡 TOTAL ROUTERS REGISTERED: {len(app.routes)}")
print("=" * 80)
print("\n")

# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Financial Stress Test Platform API",
        "version": "3.0.1",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "production"),
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "features": {
            "authentication": AUTH_LOADED,
            "stress_testing": PRODUCTION_LOADED,
            "batch_processing": PRODUCTION_LOADED,
            "risk_limits": PRODUCTION_LOADED,
            "reports": PRODUCTION_LOADED,
            "portfolio_analysis": PRODUCTION_LOADED
        },
        "endpoints": {
            "health": "/health",
            "api_health": "/api/v1/health",
            "login": "/api/v1/auth/login",
            "stress_test": "/api/v1/stress-test",
            "scenarios": "/api/v1/scenarios"
        }
    }

@app.get("/health")
async def health_check():
    """Simple health check for load balancers"""
    return {
        "status": "healthy",
        "service": "Financial Stress Test Platform",
        "version": "3.0.1",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/health")
async def detailed_health():
    """Detailed health check with component status"""
    health_status = {
        "status": "healthy",
        "service": "Financial Stress Test Platform",
        "version": "3.0.1",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "healthy",
            "auth": "loaded" if AUTH_LOADED else "not_loaded",
            "production_features": "loaded" if PRODUCTION_LOADED else "not_loaded"
        }
    }
    
    # Check GCS connectivity
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket("mlops-financial-stress-data")
        bucket.exists()
        health_status["components"]["gcs"] = "healthy"
    except Exception as e:
        health_status["components"]["gcs"] = "degraded"
        health_status["status"] = "degraded"
    
    return health_status

# ============================================================================
# ERROR HANDLERS (FIXED - RETURN JSONResponse)
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 error handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The endpoint {request.url.path} does not exist",
            "documentation": "/docs"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 error handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Check server logs."
        }
    )

# Add favicon handler to prevent 404 errors
@app.get("/favicon.ico")
async def favicon():
    """Prevent favicon 404 errors"""
    return JSONResponse(status_code=204, content={})

# ============================================================================
# RUN SERVER (LOCAL DEVELOPMENT)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Cloud Run provides PORT environment variable
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )