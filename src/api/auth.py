"""
API Authentication system.

Implements API Key authentication for the Code RAG API.

Features:
- API Key generation and validation
- Key storage (file-based or database)
- Role-based access control (admin, user, readonly)
- Key expiration and rate limiting
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
from enum import Enum

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from ..logger import get_logger


logger = get_logger(__name__)


# ============================================================================
# Models
# ============================================================================

class APIKeyRole(str, Enum):
    """API Key roles."""
    ADMIN = "admin"  # Full access including key management
    USER = "user"    # Read-write access
    READONLY = "readonly"  # Read-only access


class APIKey(BaseModel):
    """API Key model."""

    key_id: str = Field(..., description="Unique key identifier")
    key_hash: str = Field(..., description="Hashed API key")
    name: str = Field(..., description="Human-readable key name")
    role: APIKeyRole = Field(default=APIKeyRole.USER, description="Access role")
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = Field(default=None, description="Expiration time (None = never)")
    last_used_at: Optional[datetime] = None
    is_active: bool = Field(default=True, description="Is key active")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_valid(self) -> bool:
        """Check if key is valid (active and not expired)."""
        return self.is_active and not self.is_expired()


# ============================================================================
# API Key Storage
# ============================================================================

class APIKeyStore:
    """
    API Key storage.

    Stores API keys in a JSON file with hashed keys for security.
    """

    def __init__(self, storage_path: Path = Path("data/api_keys.json")):
        """
        Initialize key store.

        Args:
            storage_path: Path to JSON file for storing keys
        """
        self.storage_path = storage_path
        self.keys: Dict[str, APIKey] = {}

        # Create storage directory if needed
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing keys
        self._load_keys()

    def _load_keys(self):
        """Load keys from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)

                for key_id, key_data in data.items():
                    # Convert string dates back to datetime
                    if key_data.get('created_at'):
                        key_data['created_at'] = datetime.fromisoformat(key_data['created_at'])
                    if key_data.get('expires_at'):
                        key_data['expires_at'] = datetime.fromisoformat(key_data['expires_at'])
                    if key_data.get('last_used_at'):
                        key_data['last_used_at'] = datetime.fromisoformat(key_data['last_used_at'])

                    self.keys[key_id] = APIKey(**key_data)

                logger.info(f"Loaded {len(self.keys)} API keys from {self.storage_path}")
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")
                self.keys = {}

    def _save_keys(self):
        """Save keys to storage."""
        try:
            data = {}
            for key_id, api_key in self.keys.items():
                key_dict = api_key.model_dump()
                # Convert datetime to ISO string for JSON
                if key_dict.get('created_at'):
                    key_dict['created_at'] = key_dict['created_at'].isoformat()
                if key_dict.get('expires_at'):
                    key_dict['expires_at'] = key_dict['expires_at'].isoformat()
                if key_dict.get('last_used_at'):
                    key_dict['last_used_at'] = key_dict['last_used_at'].isoformat()

                data[key_id] = key_dict

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.keys)} API keys to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")

    @staticmethod
    def hash_key(key: str) -> str:
        """Hash an API key using SHA-256."""
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def generate_key() -> str:
        """Generate a new API key."""
        # Format: sk-rag-<random_32_chars>
        random_part = secrets.token_urlsafe(32)[:32]
        return f"sk-rag-{random_part}"

    def create_key(
        self,
        name: str,
        role: APIKeyRole = APIKeyRole.USER,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[str, APIKey]:
        """
        Create a new API key.

        Args:
            name: Human-readable key name
            role: Access role
            expires_in_days: Days until expiration (None = never)
            metadata: Additional metadata

        Returns:
            Tuple of (plain_key, APIKey object)

        Note:
            The plain key is only returned once and never stored!
        """
        # Generate key
        plain_key = self.generate_key()
        key_hash = self.hash_key(plain_key)
        key_id = secrets.token_urlsafe(16)

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        # Create APIKey object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            role=role,
            expires_at=expires_at,
            metadata=metadata or {}
        )

        # Store
        self.keys[key_id] = api_key
        self._save_keys()

        logger.info(f"Created API key: {name} (role={role}, expires={expires_at})")

        return plain_key, api_key

    def validate_key(self, plain_key: str) -> Optional[APIKey]:
        """
        Validate an API key.

        Args:
            plain_key: Plain text API key

        Returns:
            APIKey object if valid, None otherwise
        """
        key_hash = self.hash_key(plain_key)

        # Find key by hash
        for api_key in self.keys.values():
            if api_key.key_hash == key_hash:
                # Check validity
                if not api_key.is_valid():
                    logger.warning(f"Invalid API key used: {api_key.name} (expired or inactive)")
                    return None

                # Update last used
                api_key.last_used_at = datetime.now()
                self._save_keys()

                return api_key

        return None

    def get_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        return self.keys.get(key_id)

    def list_keys(self) -> List[APIKey]:
        """List all API keys."""
        return list(self.keys.values())

    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: Key ID to revoke

        Returns:
            True if revoked, False if not found
        """
        if key_id in self.keys:
            self.keys[key_id].is_active = False
            self._save_keys()
            logger.info(f"Revoked API key: {self.keys[key_id].name}")
            return True
        return False

    def delete_key(self, key_id: str) -> bool:
        """
        Delete an API key.

        Args:
            key_id: Key ID to delete

        Returns:
            True if deleted, False if not found
        """
        if key_id in self.keys:
            key_name = self.keys[key_id].name
            del self.keys[key_id]
            self._save_keys()
            logger.info(f"Deleted API key: {key_name}")
            return True
        return False


# ============================================================================
# FastAPI Security
# ============================================================================

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global key store
_key_store: Optional[APIKeyStore] = None


def get_key_store() -> APIKeyStore:
    """Get global API key store (singleton)."""
    global _key_store
    if _key_store is None:
        _key_store = APIKeyStore()
    return _key_store


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    required_role: Optional[APIKeyRole] = None
) -> APIKey:
    """
    Verify API key and optionally check role.

    Args:
        api_key: API key from header
        required_role: Required role (None = any authenticated user)

    Returns:
        APIKey object if valid

    Raises:
        HTTPException: If key is invalid or insufficient permissions
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_store = get_key_store()
    validated_key = key_store.validate_key(api_key)

    if not validated_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Check role if required
    if required_role:
        # Admin has access to everything
        if validated_key.role != APIKeyRole.ADMIN:
            # Define role hierarchy
            role_hierarchy = {
                APIKeyRole.ADMIN: 3,
                APIKeyRole.USER: 2,
                APIKeyRole.READONLY: 1,
            }

            if role_hierarchy.get(validated_key.role, 0) < role_hierarchy.get(required_role, 0):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required role: {required_role.value}",
                )

    return validated_key


# ============================================================================
# Dependency shortcuts
# ============================================================================

async def require_auth(api_key: APIKey = Security(verify_api_key)) -> APIKey:
    """Require any authenticated user."""
    return api_key


async def require_user(
    api_key: APIKey = Security(verify_api_key)
) -> APIKey:
    """Require USER or ADMIN role."""
    if api_key.role == APIKeyRole.READONLY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Read-only access. Write operations not permitted."
        )
    return api_key


async def require_admin(
    api_key: APIKey = Security(verify_api_key)
) -> APIKey:
    """Require ADMIN role."""
    if api_key.role != APIKeyRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return api_key


# ============================================================================
# Utility functions
# ============================================================================

def create_initial_admin_key() -> Optional[str]:
    """
    Create initial admin key if none exists.

    Returns:
        Plain API key if created, None if admin key already exists
    """
    key_store = get_key_store()

    # Check if admin key exists
    admin_keys = [k for k in key_store.list_keys() if k.role == APIKeyRole.ADMIN]

    if admin_keys:
        logger.info("Admin API key already exists")
        return None

    # Create admin key
    plain_key, api_key = key_store.create_key(
        name="Initial Admin Key",
        role=APIKeyRole.ADMIN,
        expires_in_days=None,  # Never expires
        metadata={"created_by": "system", "initial": True}
    )

    logger.info(f"✅ Created initial admin API key: {api_key.key_id}")
    logger.info(f"🔑 API Key: {plain_key}")
    logger.info("⚠️  SAVE THIS KEY! It will not be shown again.")

    return plain_key
