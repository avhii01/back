"""Runtime configuration, read from environment variables.

Secrets (LLM_API_KEY) live only here. They are never serialised by `public()`,
never logged (`repr=False`) and never returned by any API.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    # --- storage ---
    database_url: str = "sqlite:///./relay.db"
    sim_database_url: str = "sqlite:///./relay_sim.db"
    # --- simulated services (clearly labelled as simulated everywhere) ---
    sim_base_url: str = "http://localhost:8001"
    sim_timeout_s: float = 2.0
    # --- agent mode: auto | demo | llm ---
    agent_mode: str = "auto"
    llm_api_key: str = field(default="", repr=False)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 30.0
    # Optional USD prices per 1M tokens. Cost is only reported when both are set.
    llm_price_input_per_mtok: float | None = None
    llm_price_output_per_mtok: float | None = None
    # --- customer support / RAG ---
    # Embeddings use an OpenAI-compatible /embeddings endpoint. They fall back to
    # lexical search (clearly labelled) when no key/model or the provider is unavailable.
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = ""
    embedding_api_key: str = field(default="", repr=False)
    embedding_timeout_s: float = 20.0
    # Persistent Chroma service (Docker Compose). Empty -> DB-vector fallback.
    chroma_url: str = ""
    chroma_collection: str = "relay_policies"
    rag_min_score: float = 0.20
    rag_lexical_min_score: float = 0.34
    support_llm_enabled: bool = True
    support_max_message_chars: int = 2000
    customer_session_hours: float = 8.0
    demo_customer_auth: bool = True
    # --- durable ticket dispatch ---
    job_worker_enabled: bool = True
    job_poll_seconds: float = 0.5
    job_lease_seconds: float = 60.0
    job_max_attempts: int = 3
    # --- execution limits (enforced by deterministic code, not the model) ---
    max_steps: int = 60
    max_tool_calls: int = 60
    max_active_seconds: float = 120.0
    tool_max_attempts: int = 3
    backoff_base_s: float = 0.2
    backoff_max_s: float = 2.0
    lease_seconds: float = 60.0
    approval_ttl_seconds: float = 3600.0
    # --- web ---
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    auto_resume_on_startup: bool = True

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        values: dict = {}
        for f in dataclasses.fields(cls):
            raw = os.environ.get(f.name.upper())
            if raw is None or raw == "":
                continue
            typ = str(f.type)
            if "bool" in typ:
                values[f.name] = raw.lower() in {"1", "true", "yes"}
            elif "float" in typ:
                values[f.name] = float(raw)
            elif "int" in typ:
                values[f.name] = int(raw)
            else:
                values[f.name] = raw
        values.update(overrides)
        return cls(**values)

    def with_overrides(self, **kw) -> Settings:
        return dataclasses.replace(self, **kw)

    @property
    def mode(self) -> str:
        """Resolved agent mode: 'llm' only when a key is configured (or forced)."""
        if self.agent_mode == "demo":
            return "demo"
        if self.agent_mode == "llm":
            return "llm"
        return "llm" if self.llm_api_key else "demo"

    @property
    def embedding_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key

    @property
    def embedding_url(self) -> str:
        return (self.embedding_base_url or self.llm_base_url).rstrip("/")

    @property
    def vector_backend(self) -> str:
        return "chroma" if self.chroma_url else "database"

    def public(self) -> dict:
        """Safe-to-expose view of configuration (no secrets, no URLs with creds)."""
        return {
            "mode": self.mode,
            "model": self.llm_model if self.mode == "llm" else None,
            "llm_key_configured": bool(self.llm_api_key),
            "simulated_services": True,
            "support": {
                "enabled": True,
                "demo_customer_auth": self.demo_customer_auth,
                "rag_embedding_model": self.embedding_model if self.embedding_key else None,
                "embedding_key_configured": bool(self.embedding_key),
                "vector_backend": self.vector_backend,
                "llm_phrasing_enabled": bool(self.support_llm_enabled and self.llm_api_key),
            },
            "limits": {
                "max_steps": self.max_steps,
                "max_tool_calls": self.max_tool_calls,
                "max_active_seconds": self.max_active_seconds,
                "tool_max_attempts": self.tool_max_attempts,
            },
        }
