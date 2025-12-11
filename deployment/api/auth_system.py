# """
# ============================================================================
# MULTI-USER AUTHENTICATION & AUTHORIZATION SYSTEM - COMPLETE
# ============================================================================
# Enterprise-grade user management for Financial Stress Test Platform
# Features: JWT auth, role-based access, audit trail, session management
# ============================================================================
# """

# from fastapi import APIRouter, HTTPException, Depends, status
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from pydantic import BaseModel, EmailStr
# from typing import Optional, List, Dict
# from datetime import datetime, timedelta
# import jwt
# import bcrypt
# import sqlite3
# from pathlib import Path
# import json
# from uuid import uuid4

# router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# SECRET_KEY = "financial-stress-test-secret-key-change-in-production-2025"
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

# security = HTTPBearer()
# #DB_PATH = Path("users.db")
# DB_PATH = Path("/tmp/users.db")

# # Role permissions mapping
# ROLE_PERMISSIONS = {
#     'admin': ['all'],
#     'risk_manager': ['run_tests', 'view_all_results', 'generate_reports', 'configure_limits', 'batch_processing'],
#     'analyst': ['run_tests', 'view_own_results', 'view_reports'],
#     'viewer': ['view_results', 'view_reports'],
#     'auditor': ['view_audit_log', 'view_results', 'view_reports', 'view_all_results']
# }

# # ============================================================================
# # DATABASE INITIALIZATION
# # ============================================================================

# def init_database():
#     """Initialize SQLite database with users and audit tables"""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     # Users table
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS users (
#             user_id TEXT PRIMARY KEY,
#             email TEXT UNIQUE NOT NULL,
#             username TEXT UNIQUE NOT NULL,
#             password_hash TEXT NOT NULL,
#             role TEXT NOT NULL,
#             full_name TEXT,
#             department TEXT,
#             created_at TEXT,
#             last_login TEXT,
#             is_active INTEGER DEFAULT 1
#         )
#     """)
    
#     # Audit log table
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS audit_log (
#             log_id TEXT PRIMARY KEY,
#             user_id TEXT,
#             username TEXT,
#             action TEXT,
#             resource TEXT,
#             details TEXT,
#             timestamp TEXT,
#             ip_address TEXT
#         )
#     """)
    
#     # Create default users if database is empty
#     cursor.execute("SELECT COUNT(*) FROM users")
#     if cursor.fetchone()[0] == 0:
#         default_users = [
#             {
#                 'user_id': str(uuid4()),
#                 'email': 'admin@financialstress.com',
#                 'username': 'admin',
#                 'password': 'admin123',
#                 'role': 'admin',
#                 'full_name': 'System Administrator',
#                 'department': 'IT'
#             },
#             {
#                 'user_id': str(uuid4()),
#                 'email': 'risk.manager@financialstress.com',
#                 'username': 'risk_manager',
#                 'password': 'risk123',
#                 'role': 'risk_manager',
#                 'full_name': 'John Smith',
#                 'department': 'Risk Management'
#             },
#             {
#                 'user_id': str(uuid4()),
#                 'email': 'analyst@financialstress.com',
#                 'username': 'analyst',
#                 'password': 'analyst123',
#                 'role': 'analyst',
#                 'full_name': 'Jane Doe',
#                 'department': 'Analytics'
#             },
#             {
#                 'user_id': str(uuid4()),
#                 'email': 'viewer@financialstress.com',
#                 'username': 'viewer',
#                 'password': 'viewer123',
#                 'role': 'viewer',
#                 'full_name': 'Bob Johnson',
#                 'department': 'Executive'
#             },
#             {
#                 'user_id': str(uuid4()),
#                 'email': 'auditor@financialstress.com',
#                 'username': 'auditor',
#                 'password': 'auditor123',
#                 'role': 'auditor',
#                 'full_name': 'Sarah Williams',
#                 'department': 'Compliance'
#             }
#         ]
        
#         for user in default_users:
#             password_hash = bcrypt.hashpw(user['password'].encode(), bcrypt.gensalt()).decode()
            
#             cursor.execute("""
#                 INSERT INTO users (user_id, email, username, password_hash, role, full_name, department, created_at, is_active)
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
#             """, (user['user_id'], user['email'], user['username'], password_hash, 
#                   user['role'], user['full_name'], user['department'], 
#                   datetime.now().isoformat(), 1))
        
#         print("✅ Default users created:")
#         print("   admin / admin123 (Administrator)")
#         print("   risk_manager / risk123 (Risk Manager)")
#         print("   analyst / analyst123 (Analyst)")
#         print("   viewer / viewer123 (Viewer)")
#         print("   auditor / auditor123 (Auditor)")
    
#     conn.commit()
#     conn.close()

# # Initialize database on module import
# init_database()

# # ============================================================================
# # PYDANTIC MODELS
# # ============================================================================

# class LoginRequest(BaseModel):
#     username: str
#     password: str

# class UserCreate(BaseModel):
#     email: EmailStr
#     username: str
#     password: str
#     role: str
#     full_name: Optional[str] = None
#     department: Optional[str] = None

# class UserResponse(BaseModel):
#     user_id: str
#     email: str
#     username: str
#     role: str
#     full_name: Optional[str]
#     department: Optional[str]
#     created_at: str
#     last_login: Optional[str]
#     is_active: bool

# class TokenResponse(BaseModel):
#     access_token: str
#     token_type: str
#     user_info: Dict

# class AuditLogEntry(BaseModel):
#     user_id: str
#     action: str
#     resource: str
#     details: Optional[str] = None

# # ============================================================================
# # AUTHENTICATION UTILITIES
# # ============================================================================

# def create_access_token(data: dict, expires_delta: timedelta = None):
#     """Create JWT access token"""
#     to_encode = data.copy()
    
#     if expires_delta:
#         expire = datetime.utcnow() + expires_delta
#     else:
#         expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
#     return encoded_jwt

# def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     """Verify JWT token and return user info"""
#     try:
#         token = credentials.credentials
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
#         user_id = payload.get("sub")
#         if user_id is None:
#             raise HTTPException(status_code=401, detail="Invalid authentication token")
        
#         return payload
    
#     except jwt.ExpiredSignatureError:
#         raise HTTPException(status_code=401, detail="Token has expired. Please login again.")
#     except jwt.JWTError:
#         raise HTTPException(status_code=401, detail="Invalid authentication token")

# def require_role(required_roles: List[str]):
#     """Dependency to check if user has required role"""
#     def role_checker(current_user: dict = Depends(verify_token)):
#         if current_user['role'] not in required_roles and current_user['role'] != 'admin':
#             raise HTTPException(
#                 status_code=403,
#                 detail=f"Access denied. Required role: {', '.join(required_roles)}"
#             )
#         return current_user
#     return role_checker

# # ============================================================================
# # AUTHENTICATION ENDPOINTS
# # ============================================================================

# @router.post("/login", response_model=TokenResponse)
# async def login(request: LoginRequest):
#     """
#     User login endpoint
    
#     Returns JWT token for authenticated requests
#     """
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     # Find user
#     cursor.execute("""
#         SELECT user_id, username, email, password_hash, role, full_name, department, is_active
#         FROM users WHERE username = ?
#     """, (request.username,))
    
#     user = cursor.fetchone()
    
#     if not user:
#         conn.close()
#         raise HTTPException(status_code=401, detail="Invalid username or password")
    
#     user_id, username, email, password_hash, role, full_name, department, is_active = user
    
#     # Check if active
#     if not is_active:
#         conn.close()
#         raise HTTPException(status_code=401, detail="User account is disabled")
    
#     # Verify password
#     if not bcrypt.checkpw(request.password.encode(), password_hash.encode()):
#         conn.close()
#         raise HTTPException(status_code=401, detail="Invalid username or password")
    
#     # Update last login
#     cursor.execute("UPDATE users SET last_login = ? WHERE user_id = ?", 
#                    (datetime.now().isoformat(), user_id))
#     conn.commit()
    
#     # Create access token
#     token_data = {
#         "sub": user_id,
#         "username": username,
#         "email": email,
#         "role": role
#     }
    
#     access_token = create_access_token(token_data)
    
#     # Log login
#     log_audit_action(user_id, username, "LOGIN", "system", "User logged in successfully")
    
#     conn.close()
    
#     return {
#         "access_token": access_token,
#         "token_type": "bearer",
#         "user_info": {
#             "user_id": user_id,
#             "username": username,
#             "email": email,
#             "role": role,
#             "full_name": full_name,
#             "department": department,
#             "permissions": ROLE_PERMISSIONS.get(role, [])
#         }
#     }

# @router.post("/logout")
# async def logout(current_user: dict = Depends(verify_token)):
#     """User logout endpoint"""
#     log_audit_action(
#         current_user['sub'],
#         current_user['username'],
#         "LOGOUT",
#         "system",
#         "User logged out"
#     )
    
#     return {"message": "Logged out successfully"}

# @router.get("/me")
# async def get_current_user(current_user: dict = Depends(verify_token)):
#     """Get current user information"""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     cursor.execute("""
#         SELECT user_id, email, username, role, full_name, department, created_at, last_login
#         FROM users WHERE user_id = ?
#     """, (current_user['sub'],))
    
#     user = cursor.fetchone()
#     conn.close()
    
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")
    
#     return {
#         "user_id": user[0],
#         "email": user[1],
#         "username": user[2],
#         "role": user[3],
#         "full_name": user[4],
#         "department": user[5],
#         "created_at": user[6],
#         "last_login": user[7],
#         "permissions": ROLE_PERMISSIONS.get(user[3], [])
#     }

# # ============================================================================
# # USER MANAGEMENT ENDPOINTS (ADMIN ONLY)
# # ============================================================================

# @router.post("/users", dependencies=[Depends(require_role(['admin']))])
# async def create_user(user: UserCreate, current_user: dict = Depends(verify_token)):
#     """Create new user (admin only)"""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     # Check if username exists
#     cursor.execute("SELECT user_id FROM users WHERE username = ?", (user.username,))
#     if cursor.fetchone():
#         conn.close()
#         raise HTTPException(status_code=400, detail="Username already exists")
    
#     # Check if email exists
#     cursor.execute("SELECT user_id FROM users WHERE email = ?", (user.email,))
#     if cursor.fetchone():
#         conn.close()
#         raise HTTPException(status_code=400, detail="Email already exists")
    
#     # Validate role
#     if user.role not in ROLE_PERMISSIONS:
#         conn.close()
#         raise HTTPException(status_code=400, detail=f"Invalid role. Must be: {list(ROLE_PERMISSIONS.keys())}")
    
#     # Hash password
#     password_hash = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt()).decode()
    
#     # Create user
#     user_id = str(uuid4())
#     cursor.execute("""
#         INSERT INTO users (user_id, email, username, password_hash, role, full_name, department, created_at, is_active)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
#     """, (user_id, user.email, user.username, password_hash, user.role, 
#           user.full_name, user.department, datetime.now().isoformat(), 1))
    
#     conn.commit()
#     conn.close()
    
#     # Log action
#     log_audit_action(
#         current_user['sub'],
#         current_user['username'],
#         "USER_CREATED",
#         f"user_{user_id}",
#         f"Created user: {user.username} with role: {user.role}"
#     )
    
#     return {
#         "message": "User created successfully",
#         "user_id": user_id,
#         "username": user.username,
#         "role": user.role
#     }

# @router.get("/users", dependencies=[Depends(require_role(['admin', 'auditor']))])
# async def list_users():
#     """List all users (admin/auditor only)"""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     cursor.execute("""
#         SELECT user_id, email, username, role, full_name, department, created_at, last_login, is_active
#         FROM users ORDER BY created_at DESC
#     """)
    
#     users = []
#     for row in cursor.fetchall():
#         users.append({
#             "user_id": row[0],
#             "email": row[1],
#             "username": row[2],
#             "role": row[3],
#             "full_name": row[4],
#             "department": row[5],
#             "created_at": row[6],
#             "last_login": row[7],
#             "is_active": bool(row[8])
#         })
    
#     conn.close()
    
#     return {"users": users, "total": len(users)}

# @router.delete("/users/{user_id}", dependencies=[Depends(require_role(['admin']))])
# async def delete_user(user_id: str, current_user: dict = Depends(verify_token)):
#     """Deactivate user (admin only)"""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     # Get username before deactivating
#     cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
#     result = cursor.fetchone()
#     if not result:
#         conn.close()
#         raise HTTPException(status_code=404, detail="User not found")
    
#     deactivated_username = result[0]
    
#     # Soft delete
#     cursor.execute("UPDATE users SET is_active = 0 WHERE user_id = ?", (user_id,))
#     conn.commit()
#     conn.close()
    
#     # Log action
#     log_audit_action(
#         current_user['sub'],
#         current_user['username'],
#         "USER_DEACTIVATED",
#         f"user_{user_id}",
#         f"Deactivated user: {deactivated_username}"
#     )
    
#     return {"message": "User deactivated successfully"}

# @router.put("/users/{user_id}/activate", dependencies=[Depends(require_role(['admin']))])
# async def activate_user(user_id: str, current_user: dict = Depends(verify_token)):
#     """Reactivate user (admin only)"""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     cursor.execute("UPDATE users SET is_active = 1 WHERE user_id = ?", (user_id,))
    
#     if cursor.rowcount == 0:
#         conn.close()
#         raise HTTPException(status_code=404, detail="User not found")
    
#     conn.commit()
#     conn.close()
    
#     return {"message": "User activated successfully"}

# # ============================================================================
# # AUDIT LOG ENDPOINTS
# # ============================================================================

# @router.get("/audit-log", dependencies=[Depends(require_role(['admin', 'auditor']))])
# async def get_audit_log(limit: int = 100, user_filter: Optional[str] = None):
#     """Get audit log (admin/auditor only)"""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     if user_filter:
#         cursor.execute("""
#             SELECT log_id, user_id, username, action, resource, details, timestamp, ip_address
#             FROM audit_log WHERE username = ? ORDER BY timestamp DESC LIMIT ?
#         """, (user_filter, limit))
#     else:
#         cursor.execute("""
#             SELECT log_id, user_id, username, action, resource, details, timestamp, ip_address
#             FROM audit_log ORDER BY timestamp DESC LIMIT ?
#         """, (limit,))
    
#     logs = []
#     for row in cursor.fetchall():
#         logs.append({
#             "log_id": row[0],
#             "user_id": row[1],
#             "username": row[2],
#             "action": row[3],
#             "resource": row[4],
#             "details": row[5],
#             "timestamp": row[6],
#             "ip_address": row[7]
#         })
    
#     conn.close()
    
#     return {"total": len(logs), "entries": logs}

# @router.post("/audit-log/record")
# async def record_audit_action(entry: AuditLogEntry, current_user: dict = Depends(verify_token)):
#     """Record user action in audit log"""
#     log_audit_action(
#         current_user['sub'],
#         current_user['username'],
#         entry.action,
#         entry.resource,
#         entry.details
#     )
    
#     return {"status": "logged", "timestamp": datetime.now().isoformat()}

# # ============================================================================
# # USER STATISTICS
# # ============================================================================

# @router.get("/stats/users", dependencies=[Depends(require_role(['admin']))])
# async def get_user_statistics():
#     """Get user statistics (admin only)"""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     # Total active users
#     cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
#     total_active = cursor.fetchone()[0]
    
#     # Users by role
#     cursor.execute("SELECT role, COUNT(*) FROM users WHERE is_active = 1 GROUP BY role")
#     by_role = {row[0]: row[1] for row in cursor.fetchall()}
    
#     # Recent activity
#     cursor.execute("""
#         SELECT COUNT(*) FROM audit_log 
#         WHERE timestamp >= datetime('now', '-1 day')
#     """)
#     actions_today = cursor.fetchone()[0]
    
#     # Most active users today
#     cursor.execute("""
#         SELECT username, COUNT(*) as action_count
#         FROM audit_log 
#         WHERE timestamp >= datetime('now', '-1 day')
#         GROUP BY username
#         ORDER BY action_count DESC
#         LIMIT 5
#     """)
#     most_active = [{"username": row[0], "actions": row[1]} for row in cursor.fetchall()]
    
#     conn.close()
    
#     return {
#         "total_active_users": total_active,
#         "users_by_role": by_role,
#         "actions_today": actions_today,
#         "most_active_users_today": most_active
#     }

# # ============================================================================
# # HELPER FUNCTIONS
# # ============================================================================

# def log_audit_action(user_id: str, username: str, action: str, resource: str, details: str = None, ip: str = None):
#     """Log action to audit trail"""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     log_id = str(uuid4())
    
#     cursor.execute("""
#         INSERT INTO audit_log (log_id, user_id, username, action, resource, details, timestamp, ip_address)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#     """, (log_id, user_id, username, action, resource, details, datetime.now().isoformat(), ip))
    
#     conn.commit()
#     conn.close()

# def get_user_permissions(role: str) -> List[str]:
#     """Get permissions for a role"""
#     return ROLE_PERMISSIONS.get(role, [])

# def check_permission(user: dict, required_permission: str) -> bool:
#     """Check if user has specific permission"""
#     permissions = get_user_permissions(user['role'])
#     return 'all' in permissions or required_permission in permissions

# # ============================================================================
# # EXPORT FOR OTHER MODULES
# # ============================================================================

# __all__ = ['verify_token', 'require_role', 'log_audit_action', 'check_permission', 'router']



"""
============================================================================
MULTI-USER AUTHENTICATION & AUTHORIZATION SYSTEM - CLOUD RUN READY
============================================================================
Enterprise-grade user management for Financial Stress Test Platform
Features: JWT auth, role-based access, audit trail, session management
============================================================================
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import jwt
import bcrypt
import sqlite3
from pathlib import Path
import json
from uuid import uuid4
import os

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

# ============================================================================
# CONFIGURATION
# ============================================================================

SECRET_KEY = os.getenv("SECRET_KEY", "financial-stress-test-secret-key-change-in-production-2025")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

security = HTTPBearer()

# ✅ FIXED FOR CLOUD RUN - Use /tmp (writable directory)
DB_PATH = Path("/tmp/users.db")

# Role permissions mapping
ROLE_PERMISSIONS = {
    'admin': ['all'],
    'risk_manager': ['run_tests', 'view_all_results', 'generate_reports', 'configure_limits', 'batch_processing'],
    'analyst': ['run_tests', 'view_own_results', 'view_reports'],
    'viewer': ['view_results', 'view_reports'],
    'auditor': ['view_audit_log', 'view_results', 'view_reports', 'view_all_results']
}

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

_db_initialized = False

def init_database():
    """Initialize SQLite database with users and audit tables"""
    global _db_initialized
    
    if _db_initialized:
        return
    
    try:
        # Ensure /tmp directory exists and is writable
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                full_name TEXT,
                department TEXT,
                created_at TEXT,
                last_login TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # Audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id TEXT PRIMARY KEY,
                user_id TEXT,
                username TEXT,
                action TEXT,
                resource TEXT,
                details TEXT,
                timestamp TEXT,
                ip_address TEXT
            )
        """)
        
        # Create default users if database is empty
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            default_users = [
                {
                    'user_id': str(uuid4()),
                    'email': 'admin@financialstress.com',
                    'username': 'admin',
                    'password': 'admin123',
                    'role': 'admin',
                    'full_name': 'System Administrator',
                    'department': 'IT'
                },
                {
                    'user_id': str(uuid4()),
                    'email': 'risk.manager@financialstress.com',
                    'username': 'risk_manager',
                    'password': 'risk123',
                    'role': 'risk_manager',
                    'full_name': 'John Smith',
                    'department': 'Risk Management'
                },
                {
                    'user_id': str(uuid4()),
                    'email': 'analyst@financialstress.com',
                    'username': 'analyst',
                    'password': 'analyst123',
                    'role': 'analyst',
                    'full_name': 'Jane Doe',
                    'department': 'Analytics'
                },
                {
                    'user_id': str(uuid4()),
                    'email': 'viewer@financialstress.com',
                    'username': 'viewer',
                    'password': 'viewer123',
                    'role': 'viewer',
                    'full_name': 'Bob Johnson',
                    'department': 'Executive'
                },
                {
                    'user_id': str(uuid4()),
                    'email': 'auditor@financialstress.com',
                    'username': 'auditor',
                    'password': 'auditor123',
                    'role': 'auditor',
                    'full_name': 'Sarah Williams',
                    'department': 'Compliance'
                }
            ]
            
            for user in default_users:
                password_hash = bcrypt.hashpw(user['password'].encode(), bcrypt.gensalt()).decode()
                
                cursor.execute("""
                    INSERT INTO users (user_id, email, username, password_hash, role, full_name, department, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user['user_id'], user['email'], user['username'], password_hash, 
                      user['role'], user['full_name'], user['department'], 
                      datetime.now().isoformat(), 1))
            
            print("✅ Default users created:")
            print("   admin / admin123 (Administrator)")
            print("   risk_manager / risk123 (Risk Manager)")
            print("   analyst / analyst123 (Analyst)")
            print("   viewer / viewer123 (Viewer)")
            print("   auditor / auditor123 (Auditor)")
        
        conn.commit()
        conn.close()
        
        _db_initialized = True
        print(f"✅ Auth database initialized at {DB_PATH}")
        
    except Exception as e:
        print(f"❌ Auth database initialization failed: {e}")
        print(f"   Database path: {DB_PATH}")
        import traceback
        traceback.print_exc()
        raise

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class LoginRequest(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    role: str
    full_name: Optional[str] = None
    department: Optional[str] = None

class UserResponse(BaseModel):
    user_id: str
    email: str
    username: str
    role: str
    full_name: Optional[str]
    department: Optional[str]
    created_at: str
    last_login: Optional[str]
    is_active: bool

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_info: Dict

class AuditLogEntry(BaseModel):
    user_id: str
    action: str
    resource: str
    details: Optional[str] = None

# ============================================================================
# AUTHENTICATION UTILITIES
# ============================================================================

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        return payload
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please login again.")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

def require_role(required_roles: List[str]):
    """Dependency to check if user has required role"""
    def role_checker(current_user: dict = Depends(verify_token)):
        if current_user['role'] not in required_roles and current_user['role'] != 'admin':
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required role: {', '.join(required_roles)}"
            )
        return current_user
    return role_checker

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    User login endpoint
    
    Returns JWT token for authenticated requests
    """
    # Initialize database on first login if not already done
    if not _db_initialized:
        init_database()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Find user
    cursor.execute("""
        SELECT user_id, username, email, password_hash, role, full_name, department, is_active
        FROM users WHERE username = ?
    """, (request.username,))
    
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    user_id, username, email, password_hash, role, full_name, department, is_active = user
    
    # Check if active
    if not is_active:
        conn.close()
        raise HTTPException(status_code=401, detail="User account is disabled")
    
    # Verify password
    if not bcrypt.checkpw(request.password.encode(), password_hash.encode()):
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Update last login
    cursor.execute("UPDATE users SET last_login = ? WHERE user_id = ?", 
                   (datetime.now().isoformat(), user_id))
    conn.commit()
    
    # Create access token
    token_data = {
        "sub": user_id,
        "username": username,
        "email": email,
        "role": role
    }
    
    access_token = create_access_token(token_data)
    
    # Log login
    log_audit_action(user_id, username, "LOGIN", "system", "User logged in successfully")
    
    conn.close()
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_info": {
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "full_name": full_name,
            "department": department,
            "permissions": ROLE_PERMISSIONS.get(role, [])
        }
    }

@router.post("/logout")
async def logout(current_user: dict = Depends(verify_token)):
    """User logout endpoint"""
    log_audit_action(
        current_user['sub'],
        current_user['username'],
        "LOGOUT",
        "system",
        "User logged out"
    )
    
    return {"message": "Logged out successfully"}

@router.get("/me")
async def get_current_user(current_user: dict = Depends(verify_token)):
    """Get current user information"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT user_id, email, username, role, full_name, department, created_at, last_login
        FROM users WHERE user_id = ?
    """, (current_user['sub'],))
    
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user_id": user[0],
        "email": user[1],
        "username": user[2],
        "role": user[3],
        "full_name": user[4],
        "department": user[5],
        "created_at": user[6],
        "last_login": user[7],
        "permissions": ROLE_PERMISSIONS.get(user[3], [])
    }

# ============================================================================
# USER MANAGEMENT ENDPOINTS (ADMIN ONLY)
# ============================================================================

@router.post("/users", dependencies=[Depends(require_role(['admin']))])
async def create_user(user: UserCreate, current_user: dict = Depends(verify_token)):
    """Create new user (admin only)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if username exists
    cursor.execute("SELECT user_id FROM users WHERE username = ?", (user.username,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email exists
    cursor.execute("SELECT user_id FROM users WHERE email = ?", (user.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Validate role
    if user.role not in ROLE_PERMISSIONS:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be: {list(ROLE_PERMISSIONS.keys())}")
    
    # Hash password
    password_hash = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt()).decode()
    
    # Create user
    user_id = str(uuid4())
    cursor.execute("""
        INSERT INTO users (user_id, email, username, password_hash, role, full_name, department, created_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, user.email, user.username, password_hash, user.role, 
          user.full_name, user.department, datetime.now().isoformat(), 1))
    
    conn.commit()
    conn.close()
    
    # Log action
    log_audit_action(
        current_user['sub'],
        current_user['username'],
        "USER_CREATED",
        f"user_{user_id}",
        f"Created user: {user.username} with role: {user.role}"
    )
    
    return {
        "message": "User created successfully",
        "user_id": user_id,
        "username": user.username,
        "role": user.role
    }

@router.get("/users", dependencies=[Depends(require_role(['admin', 'auditor']))])
async def list_users():
    """List all users (admin/auditor only)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT user_id, email, username, role, full_name, department, created_at, last_login, is_active
        FROM users ORDER BY created_at DESC
    """)
    
    users = []
    for row in cursor.fetchall():
        users.append({
            "user_id": row[0],
            "email": row[1],
            "username": row[2],
            "role": row[3],
            "full_name": row[4],
            "department": row[5],
            "created_at": row[6],
            "last_login": row[7],
            "is_active": bool(row[8])
        })
    
    conn.close()
    
    return {"users": users, "total": len(users)}

@router.delete("/users/{user_id}", dependencies=[Depends(require_role(['admin']))])
async def delete_user(user_id: str, current_user: dict = Depends(verify_token)):
    """Deactivate user (admin only)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get username before deactivating
    cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    deactivated_username = result[0]
    
    # Soft delete
    cursor.execute("UPDATE users SET is_active = 0 WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    
    # Log action
    log_audit_action(
        current_user['sub'],
        current_user['username'],
        "USER_DEACTIVATED",
        f"user_{user_id}",
        f"Deactivated user: {deactivated_username}"
    )
    
    return {"message": "User deactivated successfully"}

@router.put("/users/{user_id}/activate", dependencies=[Depends(require_role(['admin']))])
async def activate_user(user_id: str, current_user: dict = Depends(verify_token)):
    """Reactivate user (admin only)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("UPDATE users SET is_active = 1 WHERE user_id = ?", (user_id,))
    
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    conn.commit()
    conn.close()
    
    return {"message": "User activated successfully"}

# ============================================================================
# AUDIT LOG ENDPOINTS
# ============================================================================

@router.get("/audit-log", dependencies=[Depends(require_role(['admin', 'auditor']))])
async def get_audit_log(limit: int = 100, user_filter: Optional[str] = None):
    """Get audit log (admin/auditor only)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if user_filter:
        cursor.execute("""
            SELECT log_id, user_id, username, action, resource, details, timestamp, ip_address
            FROM audit_log WHERE username = ? ORDER BY timestamp DESC LIMIT ?
        """, (user_filter, limit))
    else:
        cursor.execute("""
            SELECT log_id, user_id, username, action, resource, details, timestamp, ip_address
            FROM audit_log ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
    
    logs = []
    for row in cursor.fetchall():
        logs.append({
            "log_id": row[0],
            "user_id": row[1],
            "username": row[2],
            "action": row[3],
            "resource": row[4],
            "details": row[5],
            "timestamp": row[6],
            "ip_address": row[7]
        })
    
    conn.close()
    
    return {"total": len(logs), "entries": logs}

@router.post("/audit-log/record")
async def record_audit_action(entry: AuditLogEntry, current_user: dict = Depends(verify_token)):
    """Record user action in audit log"""
    log_audit_action(
        current_user['sub'],
        current_user['username'],
        entry.action,
        entry.resource,
        entry.details
    )
    
    return {"status": "logged", "timestamp": datetime.now().isoformat()}

# ============================================================================
# USER STATISTICS
# ============================================================================

@router.get("/stats/users", dependencies=[Depends(require_role(['admin']))])
async def get_user_statistics():
    """Get user statistics (admin only)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total active users
    cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
    total_active = cursor.fetchone()[0]
    
    # Users by role
    cursor.execute("SELECT role, COUNT(*) FROM users WHERE is_active = 1 GROUP BY role")
    by_role = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Recent activity
    cursor.execute("""
        SELECT COUNT(*) FROM audit_log 
        WHERE timestamp >= datetime('now', '-1 day')
    """)
    actions_today = cursor.fetchone()[0]
    
    # Most active users today
    cursor.execute("""
        SELECT username, COUNT(*) as action_count
        FROM audit_log 
        WHERE timestamp >= datetime('now', '-1 day')
        GROUP BY username
        ORDER BY action_count DESC
        LIMIT 5
    """)
    most_active = [{"username": row[0], "actions": row[1]} for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "total_active_users": total_active,
        "users_by_role": by_role,
        "actions_today": actions_today,
        "most_active_users_today": most_active
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log_audit_action(user_id: str, username: str, action: str, resource: str, details: str = None, ip: str = None):
    """Log action to audit trail"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    log_id = str(uuid4())
    
    cursor.execute("""
        INSERT INTO audit_log (log_id, user_id, username, action, resource, details, timestamp, ip_address)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (log_id, user_id, username, action, resource, details, datetime.now().isoformat(), ip))
    
    conn.commit()
    conn.close()

def get_user_permissions(role: str) -> List[str]:
    """Get permissions for a role"""
    return ROLE_PERMISSIONS.get(role, [])

def check_permission(user: dict, required_permission: str) -> bool:
    """Check if user has specific permission"""
    permissions = get_user_permissions(user['role'])
    return 'all' in permissions or required_permission in permissions

# ============================================================================
# INITIALIZE ON MODULE LOAD (WITH ERROR HANDLING)
# ============================================================================

try:
    init_database()
    print("✅ Auth system ready: Database initialized on startup")
except Exception as e:
    print(f"⚠️  Auth database initialization deferred: {e}")
    print(f"   Will initialize on first login attempt")

# ============================================================================
# EXPORT FOR OTHER MODULES
# ============================================================================

__all__ = ['verify_token', 'require_role', 'log_audit_action', 'check_permission', 'router']