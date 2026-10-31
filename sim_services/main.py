import os

from sim_services.app import create_sim_app

app = create_sim_app(os.environ.get("SIM_DATABASE_URL", "sqlite:///./relay_sim.db"))
