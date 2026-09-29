"""API-key auth middleware.

Keys come from LNM_API_KEYS (comma-separated). The Authorization header must be
"Bearer <key>". Admin routes additionally require the admin key
(LNM_ADMIN_KEY) or any valid API key when no admin key is configured.

For tests: GatewayConfig can be constructed with explicit keys.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field


@dataclass
class GatewayConfig:
    api_keys: set[str] = field(default_factory=set)
    admin_key: str = ""
    rate_limit_per_min: int = 60
    backend_name: str = "ollama"
    audit_path: str = "audit.jsonl"
    cost_per_1k_tokens_usd: float = 0.0015  # documented estimate, see README

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        keys = {
            k.strip() for k in os.environ.get("LNM_API_KEYS", "").split(",") if k.strip()
        }
        return cls(
            api_keys=keys,
            admin_key=os.environ.get("LNM_ADMIN_KEY", ""),
            rate_limit_per_min=int(os.environ.get("LNM_RATE_LIMIT_PER_MIN", "60")),
            backend_name=os.environ.get("LNM_BACKEND", "ollama"),
            audit_path=os.environ.get("LNM_AUDIT_PATH", "audit.jsonl"),
            cost_per_1k_tokens_usd=float(
                os.environ.get("LNM_COST_PER_1K_USD", "0.0015")),
        )

    def is_valid_key(self, key: str) -> bool:
        return bool(key) and key in self.api_keys

    def is_admin(self, key: str) -> bool:
        if self.admin_key:
            return key == self.admin_key
        return self.is_valid_key(key)


def key_hash(key: str) -> str:
    """Hash for audit logs — never log raw API keys."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]
