import os

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import inspect

from relay.db import make_engine
from relay.models import Base


def test_migration_creates_schema_matching_models(tmp_path, monkeypatch):
    url = os.environ.get("RELAY_TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'mig.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    engine = make_engine(url)
    if url.startswith("postgres"):
        from sqlalchemy import text
        with engine.begin() as c:
            c.execute(text("DROP SCHEMA public CASCADE"))
            c.execute(text("CREATE SCHEMA public"))
    cfg = Config(os.path.join(os.path.dirname(__file__), "..", "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "migrations"))
    command.upgrade(cfg, "head")
    tables = set(inspect(engine).get_table_names())
    assert set(Base.metadata.tables) <= tables
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], f"models and migrations have drifted: {diff}"
    command.downgrade(cfg, "base")
    assert not (set(Base.metadata.tables) & set(inspect(engine).get_table_names()))
