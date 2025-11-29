"""
API Key management endpoints.

Provides:
- POST /api/keys - Create new API key (admin only)
- GET /api/keys - List all API keys (admin only)
- DELETE /api/keys/{key_id} - Delete API key (admin only)
- POST /api/keys/{key_id}/revoke - Revoke API key (admin only)
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime

from ..auth import (
    APIKey,
    APIKeyRole,
    APIKeyStore,
    get_key_store,
    require_admin,
)
from ...logger import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/api/keys", tags=["authentication"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str = Field(..., description="Human-readable key name", min_length=1)
    role: APIKeyRole = Field(default=APIKeyRole.USER, description="Access role")
    expires_in_days: int | None = Field(
        default=None,
        description="Days until expiration (None = never)",
        ge=1,
        le=365
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Telegram Bot Key",
                "role": "user",
                "expires_in_days": 90,
                "metadata": {"service": "telegram"}
            }
        }


class CreateKeyResponse(BaseModel):
    """Response when creating a new API key."""

    api_key: str = Field(..., description="Plain API key (SAVE THIS!)")
    key_id: str = Field(..., description="Key ID")
    name: str = Field(..., description="Key name")
    role: APIKeyRole = Field(..., description="Access role")
    created_at: datetime
    expires_at: datetime | None

    warning: str = Field(
        default="⚠️ SAVE THIS KEY! It will not be shown again.",
        description="Security warning"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "api_key": "sk-rag-abc123xyz789...",
                "key_id": "key_abc123",
                "name": "Telegram Bot Key",
                "role": "user",
                "created_at": "2025-11-29T12:00:00Z",
                "expires_at": "2026-02-27T12:00:00Z",
                "warning": "⚠️ SAVE THIS KEY! It will not be shown again."
            }
        }


class APIKeyInfo(BaseModel):
    """API Key information (without the actual key)."""

    key_id: str
    name: str
    role: APIKeyRole
    created_at: datetime
    expires_at: datetime | None
    last_used_at: datetime | None
    is_active: bool
    is_expired: bool
    metadata: dict

    class Config:
        json_schema_extra = {
            "example": {
                "key_id": "key_abc123",
                "name": "Telegram Bot Key",
                "role": "user",
                "created_at": "2025-11-29T12:00:00Z",
                "expires_at": "2026-02-27T12:00:00Z",
                "last_used_at": "2025-11-29T12:30:00Z",
                "is_active": True,
                "is_expired": False,
                "metadata": {"service": "telegram"}
            }
        }


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "",
    response_model=CreateKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
    description="Create a new API key (admin only)"
)
async def create_key(
    request: CreateKeyRequest,
    key_store: APIKeyStore = Depends(get_key_store),
    _current_key: APIKey = Depends(require_admin),
):
    """
    Create a new API key.

    **Requires admin access.**

    Returns the plain API key which will NOT be stored or shown again.
    Save it immediately!

    **Example:**
    ```json
    {
        "name": "Telegram Bot Key",
        "role": "user",
        "expires_in_days": 90
    }
    ```
    """
    try:
        # Create key
        plain_key, api_key = key_store.create_key(
            name=request.name,
            role=request.role,
            expires_in_days=request.expires_in_days,
            metadata=request.metadata
        )

        logger.info(f"Admin '{_current_key.name}' created new API key: {request.name}")

        return CreateKeyResponse(
            api_key=plain_key,
            key_id=api_key.key_id,
            name=api_key.name,
            role=api_key.role,
            created_at=api_key.created_at,
            expires_at=api_key.expires_at,
        )

    except Exception as e:
        logger.error(f"Failed to create API key: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}"
        )


@router.get(
    "",
    response_model=List[APIKeyInfo],
    summary="List API keys",
    description="List all API keys (admin only)"
)
async def list_keys(
    key_store: APIKeyStore = Depends(get_key_store),
    _current_key: APIKey = Depends(require_admin),
):
    """
    List all API keys.

    **Requires admin access.**

    Returns information about all keys (without the actual key values).
    """
    keys = key_store.list_keys()

    return [
        APIKeyInfo(
            key_id=key.key_id,
            name=key.name,
            role=key.role,
            created_at=key.created_at,
            expires_at=key.expires_at,
            last_used_at=key.last_used_at,
            is_active=key.is_active,
            is_expired=key.is_expired(),
            metadata=key.metadata,
        )
        for key in keys
    ]


@router.post(
    "/{key_id}/revoke",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API key",
    description="Revoke an API key (admin only)"
)
async def revoke_key(
    key_id: str,
    key_store: APIKeyStore = Depends(get_key_store),
    _current_key: APIKey = Depends(require_admin),
):
    """
    Revoke an API key.

    **Requires admin access.**

    The key will be marked as inactive but not deleted.
    """
    success = key_store.revoke_key(key_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}"
        )

    logger.info(f"Admin '{_current_key.name}' revoked API key: {key_id}")


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete API key",
    description="Permanently delete an API key (admin only)"
)
async def delete_key(
    key_id: str,
    key_store: APIKeyStore = Depends(get_key_store),
    _current_key: APIKey = Depends(require_admin),
):
    """
    Delete an API key permanently.

    **Requires admin access.**

    **Warning:** This action cannot be undone!
    """
    # Prevent deleting own key
    if key_id == _current_key.key_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own API key"
        )

    success = key_store.delete_key(key_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}"
        )

    logger.info(f"Admin '{_current_key.name}' deleted API key: {key_id}")


@router.get(
    "/me",
    response_model=APIKeyInfo,
    summary="Get current key info",
    description="Get information about the current API key"
)
async def get_current_key_info(
    current_key: APIKey = Depends(require_admin),  # Actually require_auth but using admin for consistency
):
    """
    Get information about the current API key.

    This endpoint can be used to verify that your API key is valid
    and check its permissions.
    """
    return APIKeyInfo(
        key_id=current_key.key_id,
        name=current_key.name,
        role=current_key.role,
        created_at=current_key.created_at,
        expires_at=current_key.expires_at,
        last_used_at=current_key.last_used_at,
        is_active=current_key.is_active,
        is_expired=current_key.is_expired(),
        metadata=current_key.metadata,
    )
