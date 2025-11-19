"""
Security & Access Control System
Phase 1: Data Architecture Foundation

Provides comprehensive security features including data encryption,
access control, audit logging, and privacy-by-design implementation.
"""

import os
import json
import hashlib
import hmac
import secrets
import string
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import sqlite3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import re
import pandas as pd

class AccessLevel(Enum):
    """Access control levels"""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

class DataSensitivity(Enum):
    """Data sensitivity classifications"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class PrivacyRegulation(Enum):
    """Supported privacy regulations"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"

@dataclass
class User:
    """User account information"""
    user_id: str
    username: str
    email: str
    role: str
    access_level: AccessLevel
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    password_hash: Optional[str] = None

@dataclass
class AccessPolicy:
    """Access control policy"""
    policy_id: str
    name: str
    description: str
    resource_pattern: str  # Regex pattern for matching resources
    allowed_roles: List[str] = field(default_factory=list)
    allowed_users: List[str] = field(default_factory=list)
    required_access_level: AccessLevel = AccessLevel.READ
    conditions: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

@dataclass
class AuditLogEntry:
    """Audit log entry"""
    entry_id: str
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    resource_type: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

@dataclass
class DataClassification:
    """Data classification and handling rules"""
    classification_id: str
    name: str
    sensitivity: DataSensitivity
    retention_period_days: int
    encryption_required: bool
    access_logging_required: bool
    privacy_regulations: List[PrivacyRegulation] = field(default_factory=list)
    data_masks: Dict[str, str] = field(default_factory=dict)  # Field -> mask pattern

class EncryptionManager:
    """
    Manages data encryption and decryption operations
    """

    def __init__(self, key_file: str = "encryption_key.key"):
        self.key_file = Path(key_file)
        self._ensure_key_exists()

    def _ensure_key_exists(self):
        """Generate and save encryption key if it doesn't exist"""
        if not self.key_file.exists():
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(self.key_file, 0o600)

    def _load_key(self) -> bytes:
        """Load encryption key"""
        with open(self.key_file, 'rb') as f:
            return f.read()

    def encrypt_data(self, data: Union[str, bytes, dict]) -> str:
        """
        Encrypt data using Fernet symmetric encryption

        Args:
            data: Data to encrypt (string, bytes, or dict)

        Returns:
            Base64-encoded encrypted data
        """
        try:
            fernet = Fernet(self._load_key())

            # Convert data to bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data, default=str).encode()
            elif isinstance(data, str):
                data_bytes = data.encode()
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = str(data).encode()

            encrypted = fernet.encrypt(data_bytes)
            return base64.b64encode(encrypted).decode()

        except Exception as e:
            logging.error(f"Encryption failed: {str(e)}")
            raise

    def decrypt_data(self, encrypted_data: str) -> Union[str, dict]:
        """
        Decrypt data

        Args:
            encrypted_data: Base64-encoded encrypted data

        Returns:
            Decrypted data (string or dict if JSON)
        """
        try:
            fernet = Fernet(self._load_key())

            encrypted_bytes = base64.b64decode(encrypted_data)
            decrypted_bytes = fernet.decrypt(encrypted_bytes)
            decrypted_str = decrypted_bytes.decode()

            # Try to parse as JSON
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str

        except Exception as e:
            logging.error(f"Decryption failed: {str(e)}")
            raise

    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> str:
        """
        Create a secure hash of sensitive data

        Args:
            data: Data to hash
            salt: Optional salt (generated if not provided)

        Returns:
            Hexadecimal hash string
        """
        if salt is None:
            salt = secrets.token_hex(16)

        # Use PBKDF2 for key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
        )

        key = kdf.derive(data.encode())
        return f"{salt}:{key.hex()}"

    def verify_hash(self, data: str, hash_value: str) -> bool:
        """
        Verify data against a hash

        Args:
            data: Original data
            hash_value: Hash to verify against

        Returns:
            True if hash matches
        """
        try:
            salt, expected_key = hash_value.split(':', 1)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode(),
                iterations=100000,
            )
            key = kdf.derive(data.encode())
            return key.hex() == expected_key
        except Exception:
            return False

class AccessControlManager:
    """
    Manages access control and authorization
    """

    def __init__(self, db_path: str = "access_control.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._initialize_database()

    def _initialize_database(self):
        """Initialize access control database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    access_level TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    password_hash TEXT
                )
            ''')

            # Access policies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS access_policies (
                    policy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    resource_pattern TEXT NOT NULL,
                    allowed_roles TEXT,  -- JSON array
                    allowed_users TEXT,  -- JSON array
                    required_access_level TEXT NOT NULL,
                    conditions TEXT,     -- JSON object
                    is_active BOOLEAN DEFAULT 1
                )
            ''')

            # Sessions table for temporary access
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')

            conn.commit()

    def create_user(self, user: User) -> bool:
        """Create a new user account"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (user_id, username, email, role, access_level,
                                     is_active, created_at, password_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user.user_id, user.username, user.email, user.role,
                    user.access_level.value, user.is_active,
                    user.created_at.isoformat(), user.password_hash
                ))
                conn.commit()
            self.logger.info(f"Created user: {user.username}")
            return True
        except sqlite3.IntegrityError:
            self.logger.error(f"User already exists: {user.username}")
            return False
        except Exception as e:
            self.logger.error(f"Error creating user: {str(e)}")
            return False

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, username, email, role, access_level, is_active,
                           created_at, last_login, password_hash
                    FROM users WHERE username = ? AND is_active = 1
                ''', (username,))

                row = cursor.fetchone()
                if not row:
                    return None

                # Verify password hash
                stored_hash = row[8]
                if stored_hash and self._verify_password(password, stored_hash):
                    # Update last login
                    cursor.execute('''
                        UPDATE users SET last_login = ? WHERE user_id = ?
                    ''', (datetime.now().isoformat(), row[0]))
                    conn.commit()

                    return User(
                        user_id=row[0],
                        username=row[1],
                        email=row[2],
                        role=row[3],
                        access_level=AccessLevel(row[4]),
                        is_active=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        last_login=datetime.fromisoformat(row[7]) if row[7] else None,
                        password_hash=row[8]
                    )

        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")

        return None

    def check_access(self, user: User, resource: str, action: str,
                    context: Dict[str, Any] = None) -> bool:
        """
        Check if user has access to perform action on resource

        Args:
            user: User requesting access
            resource: Resource being accessed
            action: Action being performed
            context: Additional context for access decision

        Returns:
            True if access is granted
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM access_policies WHERE is_active = 1')
                policies = cursor.fetchall()

                for policy_row in policies:
                    policy = self._row_to_policy(policy_row)

                    # Check if resource matches pattern
                    if not re.match(policy.resource_pattern, resource):
                        continue

                    # Check access level
                    required_level = policy.required_access_level
                    user_level = user.access_level

                    if not self._check_access_level(user_level, required_level):
                        continue

                    # Check role-based access
                    if policy.allowed_roles and user.role not in policy.allowed_roles:
                        continue

                    # Check user-specific access
                    if policy.allowed_users and user.user_id not in policy.allowed_users:
                        continue

                    # Check additional conditions
                    if policy.conditions and not self._check_conditions(policy.conditions, context or {}):
                        continue

                    return True

            return False

        except Exception as e:
            self.logger.error(f"Access check error: {str(e)}")
            return False

    def _check_access_level(self, user_level: AccessLevel, required_level: AccessLevel) -> bool:
        """Check if user access level meets requirements"""
        level_hierarchy = {
            AccessLevel.NONE: 0,
            AccessLevel.READ: 1,
            AccessLevel.WRITE: 2,
            AccessLevel.ADMIN: 3
        }

        return level_hierarchy[user_level] >= level_hierarchy[required_level]

    def _check_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check additional access conditions"""
        for key, expected_value in conditions.items():
            if key not in context:
                return False
            if context[key] != expected_value:
                return False
        return True

    def create_access_policy(self, policy: AccessPolicy) -> bool:
        """Create a new access policy"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO access_policies
                    (policy_id, name, description, resource_pattern, allowed_roles,
                     allowed_users, required_access_level, conditions, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    policy.policy_id, policy.name, policy.description,
                    policy.resource_pattern, json.dumps(policy.allowed_roles),
                    json.dumps(policy.allowed_users), policy.required_access_level.value,
                    json.dumps(policy.conditions), policy.is_active
                ))
                conn.commit()
            self.logger.info(f"Created access policy: {policy.name}")
            return True
        except Exception as e:
            self.logger.error(f"Error creating policy: {str(e)}")
            return False

    def _hash_password(self, password: str) -> str:
        """Hash a password for storage"""
        salt = secrets.token_hex(16)
        return self._hash_with_salt(password, salt)

    def _hash_with_salt(self, password: str, salt: str) -> str:
        """Hash password with salt"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        return f"{salt}:{key.hex()}"

    def _verify_password(self, password: str, hash_value: str) -> bool:
        """Verify password against hash"""
        try:
            salt, expected_key = hash_value.split(':', 1)
            computed_hash = self._hash_with_salt(password, salt)
            return computed_hash == hash_value
        except Exception:
            return False

    def _row_to_policy(self, row) -> AccessPolicy:
        """Convert database row to AccessPolicy"""
        return AccessPolicy(
            policy_id=row[0],
            name=row[1],
            description=row[2],
            resource_pattern=row[3],
            allowed_roles=json.loads(row[4]) if row[4] else [],
            allowed_users=json.loads(row[5]) if row[5] else [],
            required_access_level=AccessLevel(row[6]),
            conditions=json.loads(row[7]) if row[7] else {},
            is_active=row[8]
        )

class AuditLogger:
    """
    Comprehensive audit logging system
    """

    def __init__(self, log_file: str = "audit.log", db_path: str = "audit.db"):
        self.log_file = Path(log_file)
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._initialize_database()

    def _initialize_database(self):
        """Initialize audit database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    details TEXT,  -- JSON
                    ip_address TEXT,
                    user_agent TEXT
                )
            ''')
            conn.commit()

    def log_access(self, entry: AuditLogEntry):
        """Log an access event"""
        try:
            # Write to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO audit_log
                    (entry_id, timestamp, user_id, action, resource, resource_type,
                     success, details, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.entry_id, entry.timestamp.isoformat(), entry.user_id,
                    entry.action, entry.resource, entry.resource_type, entry.success,
                    json.dumps(entry.details), entry.ip_address, entry.user_agent
                ))
                conn.commit()

            # Write to log file
            log_entry = {
                'timestamp': entry.timestamp.isoformat(),
                'user_id': entry.user_id,
                'action': entry.action,
                'resource': entry.resource,
                'resource_type': entry.resource_type,
                'success': entry.success,
                'ip_address': entry.ip_address,
                'user_agent': entry.user_agent,
                'details': entry.details
            }

            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            self.logger.error(f"Audit logging error: {str(e)}")

    def get_audit_trail(self, user_id: str = None, resource: str = None,
                       action: str = None, start_date: datetime = None,
                       end_date: datetime = None, limit: int = 100) -> List[AuditLogEntry]:
        """Retrieve audit trail with filters"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                conditions = []
                params = []

                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)
                if resource:
                    conditions.append("resource LIKE ?")
                    params.append(f"%{resource}%")
                if action:
                    conditions.append("action = ?")
                    params.append(action)
                if start_date:
                    conditions.append("timestamp >= ?")
                    params.append(start_date.isoformat())
                if end_date:
                    conditions.append("timestamp <= ?")
                    params.append(end_date.isoformat())

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                cursor.execute(f'''
                    SELECT * FROM audit_log
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', params + [limit])

                entries = []
                for row in cursor.fetchall():
                    entries.append(AuditLogEntry(
                        entry_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        user_id=row[2],
                        action=row[3],
                        resource=row[4],
                        resource_type=row[5],
                        success=row[6],
                        details=json.loads(row[7]) if row[7] else {},
                        ip_address=row[8],
                        user_agent=row[9]
                    ))

                return entries

        except Exception as e:
            self.logger.error(f"Error retrieving audit trail: {str(e)}")
            return []

class PrivacyManager:
    """
    Privacy-by-design implementation with GDPR/CCPA compliance
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classifications = self._load_default_classifications()

    def _load_default_classifications(self) -> Dict[str, DataClassification]:
        """Load default data classifications"""
        return {
            'public': DataClassification(
                classification_id='public',
                name='Public Data',
                sensitivity=DataSensitivity.PUBLIC,
                retention_period_days=365*7,  # 7 years
                encryption_required=False,
                access_logging_required=False
            ),
            'internal': DataClassification(
                classification_id='internal',
                name='Internal Data',
                sensitivity=DataSensitivity.INTERNAL,
                retention_period_days=365*5,  # 5 years
                encryption_required=False,
                access_logging_required=True
            ),
            'confidential': DataClassification(
                classification_id='confidential',
                name='Confidential Data',
                sensitivity=DataSensitivity.CONFIDENTIAL,
                retention_period_days=365*3,  # 3 years
                encryption_required=True,
                access_logging_required=True,
                privacy_regulations=[PrivacyRegulation.GDPR, PrivacyRegulation.CCPA],
                data_masks={
                    'email': '***@***.***',
                    'phone': '***-***-****',
                    'ssn': '***-**-****'
                }
            ),
            'restricted': DataClassification(
                classification_id='restricted',
                name='Restricted Data',
                sensitivity=DataSensitivity.RESTRICTED,
                retention_period_days=365*2,  # 2 years
                encryption_required=True,
                access_logging_required=True,
                privacy_regulations=[PrivacyRegulation.GDPR, PrivacyRegulation.CCPA, PrivacyRegulation.HIPAA],
                data_masks={
                    'email': '***@***.***',
                    'phone': '***-***-****',
                    'ssn': '***-**-****',
                    'medical_id': '***-***-****'
                }
            )
        }

    def classify_data(self, data_type: str, contains_pii: bool = False,
                     contains_health: bool = False) -> DataClassification:
        """
        Classify data based on type and content

        Args:
            data_type: Type of data (breach_data, user_logs, etc.)
            contains_pii: Whether data contains personally identifiable information
            contains_health: Whether data contains health information

        Returns:
            Appropriate data classification
        """
        if contains_health:
            return self.classifications['restricted']
        elif contains_pii:
            return self.classifications['confidential']
        elif data_type in ['breach_data', 'threat_intel']:
            return self.classifications['internal']
        else:
            return self.classifications['public']

    def apply_data_masks(self, df: pd.DataFrame, classification: DataClassification) -> pd.DataFrame:
        """
        Apply data masking based on classification rules

        Args:
            df: DataFrame to mask
            classification: Data classification with masking rules

        Returns:
            Masked DataFrame
        """
        if not classification.data_masks:
            return df

        masked_df = df.copy()

        for column, mask_pattern in classification.data_masks.items():
            if column in masked_df.columns:
                # Apply masking function based on pattern
                if 'email' in column.lower():
                    masked_df[column] = masked_df[column].apply(self._mask_email)
                elif 'phone' in column.lower():
                    masked_df[column] = masked_df[column].apply(self._mask_phone)
                elif 'ssn' in column.lower():
                    masked_df[column] = masked_df[column].apply(self._mask_ssn)
                else:
                    # Generic masking
                    masked_df[column] = mask_pattern

        return masked_df

    def _mask_email(self, email: str) -> str:
        """Mask email address"""
        if not email or '@' not in email:
            return email
        local, domain = email.split('@', 1)
        return f"{local[:2]}***@{domain}"

    def _mask_phone(self, phone: str) -> str:
        """Mask phone number"""
        if not phone:
            return phone
        # Remove non-digits
        digits = re.sub(r'\D', '', str(phone))
        if len(digits) >= 10:
            return f"***-***-{digits[-4:]}"
        return phone

    def _mask_ssn(self, ssn: str) -> str:
        """Mask Social Security Number"""
        if not ssn:
            return ssn
        digits = re.sub(r'\D', '', str(ssn))
        if len(digits) == 9:
            return f"***-**-{digits[-4:]}"
        return ssn

    def check_retention_compliance(self, data_age_days: int,
                                  classification: DataClassification) -> bool:
        """
        Check if data retention complies with classification rules

        Args:
            data_age_days: Age of data in days
            classification: Data classification

        Returns:
            True if compliant
        """
        return data_age_days <= classification.retention_period_days

class SecurityManager:
    """
    Unified security management system
    """

    def __init__(self):
        self.encryption = EncryptionManager()
        self.access_control = AccessControlManager()
        self.audit_logger = AuditLogger()
        self.privacy_manager = PrivacyManager()
        self.logger = logging.getLogger(__name__)

    def secure_data_operation(self, user: User, operation: str, resource: str,
                            data: Any = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform a secure data operation with full security controls

        Args:
            user: User performing operation
            operation: Operation type (read, write, delete, etc.)
            resource: Resource being accessed
            data: Data involved in operation
            context: Additional context

        Returns:
            Operation result with security metadata
        """
        start_time = datetime.now()
        success = False
        error_message = ""

        try:
            # Check access permissions
            if not self.access_control.check_access(user, resource, operation, context):
                error_message = "Access denied"
                return {
                    'success': False,
                    'error': error_message,
                    'data': None
                }

            # Perform operation based on type
            result_data = None
            if operation == 'read':
                result_data = self._secure_read(resource, user)
            elif operation == 'write':
                result_data = self._secure_write(resource, data, user)
            elif operation == 'delete':
                result_data = self._secure_delete(resource, user)
            else:
                error_message = f"Unsupported operation: {operation}"

            success = error_message == ""

            return {
                'success': success,
                'error': error_message if not success else None,
                'data': result_data
            }

        except Exception as e:
            error_message = str(e)
            self.logger.error(f"Security operation failed: {error_message}")
            return {
                'success': False,
                'error': error_message,
                'data': None
            }

        finally:
            # Log the operation
            audit_entry = AuditLogEntry(
                entry_id=secrets.token_hex(16),
                timestamp=start_time,
                user_id=user.user_id,
                action=operation,
                resource=resource,
                resource_type=self._get_resource_type(resource),
                success=success,
                details={
                    'error_message': error_message,
                    'execution_time': (datetime.now() - start_time).total_seconds(),
                    'context': context or {}
                }
            )
            self.audit_logger.log_access(audit_entry)

    def _secure_read(self, resource: str, user: User) -> Any:
        """Perform secure read operation"""
        # Implementation would depend on resource type
        # For now, return placeholder
        return {"message": f"Secure read from {resource}"}

    def _secure_write(self, resource: str, data: Any, user: User) -> Any:
        """Perform secure write operation"""
        # Implementation would depend on resource type
        # For now, return placeholder
        return {"message": f"Secure write to {resource}"}

    def _secure_delete(self, resource: str, user: User) -> Any:
        """Perform secure delete operation"""
        # Implementation would depend on resource type
        # For now, return placeholder
        return {"message": f"Secure delete of {resource}"}

    def _get_resource_type(self, resource: str) -> str:
        """Determine resource type from resource string"""
        if 'dataset' in resource:
            return 'dataset'
        elif 'model' in resource:
            return 'model'
        elif 'config' in resource:
            return 'configuration'
        else:
            return 'unknown'

# Convenience functions
def create_security_manager() -> SecurityManager:
    """Create a security manager instance"""
    return SecurityManager()

def encrypt_sensitive_data(data: Union[str, dict]) -> str:
    """Encrypt sensitive data"""
    manager = EncryptionManager()
    return manager.encrypt_data(data)

def decrypt_sensitive_data(encrypted_data: str) -> Union[str, dict]:
    """Decrypt sensitive data"""
    manager = EncryptionManager()
    return manager.decrypt_data(encrypted_data)

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize security manager
    security = SecurityManager()

    # Create a test user
    test_user = User(
        user_id="user_001",
        username="analyst",
        email="analyst@company.com",
        role="data_analyst",
        access_level=AccessLevel.READ
    )

    # Create user account
    security.access_control.create_user(test_user)

    # Test secure operation
    result = security.secure_data_operation(
        user=test_user,
        operation="read",
        resource="dataset:breach_data",
        context={"department": "security"}
    )

    print(f"Operation result: {result}")
