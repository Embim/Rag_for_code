"""
Base configuration class with common functionality.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import os
import yaml


@dataclass
class BaseConfig:
    """
    Base class for all configuration dataclasses.
    
    Provides:
    - to_dict() conversion
    - from_dict() factory
    - from_env() environment loading
    - from_yaml() file loading
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary."""
        # Filter only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    @classmethod
    def from_env(cls, prefix: str = "") -> 'BaseConfig':
        """
        Create config from environment variables.
        
        Args:
            prefix: Environment variable prefix (e.g., "RAG_")
        """
        data = {}
        for field_name, field_info in cls.__dataclass_fields__.items():
            env_name = f"{prefix}{field_name.upper()}"
            env_value = os.getenv(env_name)
            
            if env_value is not None:
                # Type conversion
                field_type = field_info.type
                if field_type == bool:
                    data[field_name] = env_value.lower() in ('true', '1', 'yes')
                elif field_type == int:
                    data[field_name] = int(env_value)
                elif field_type == float:
                    data[field_name] = float(env_value)
                else:
                    data[field_name] = env_value
        
        return cls(**data) if data else cls()
    
    @classmethod
    def from_yaml(cls, path: str, section: Optional[str] = None) -> 'BaseConfig':
        """
        Load config from YAML file.
        
        Args:
            path: Path to YAML file
            section: Optional section name within the file
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if section and section in data:
            data = data[section]
        
        return cls.from_dict(data)
    
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """
        Merge with another config, other's values take precedence.
        """
        merged = self.to_dict()
        merged.update({k: v for k, v in other.to_dict().items() if v is not None})
        return self.__class__.from_dict(merged)

