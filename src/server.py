import os
import uvicorn
import threading
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import hydra
from omegaconf import DictConfig, OmegaConf

from src.experiment import Experiment
from src.utils import deep_merge

logger = logging.getLogger(__name__)

app = FastAPI(title="ACID_sbs API", description="Prototypical learning experiment server")

server_args = {}          # populated by run_server() from the Hydra config
experiment = None         # current Experiment instance
experiment_thread = None  # background training thread
experiment_lock = threading.Lock()


def setup_logging(level_str: str = "INFO"):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


@app.get("/")
async def root():
    return {"msg": "ACID_sbs server is running"}


@app.post("/start")
async def start(init_args: dict = {}):
    """
    Initialises and starts the experiment in one call.
    The optional JSON body is deep-merged on top of the YAML config,
    so the caller can override any field at runtime without touching files.
    If an experiment is already running, returns 400.
    """
    global experiment, experiment_thread
    with experiment_lock:
        if experiment_thread is not None and experiment_thread.is_alive():
            logger.warning("Received /start but experiment is already running")
            return JSONResponse(
                status_code=400,
                content={"error": "Experiment already running. Call /stop first."},
            )
        try:
            merged_args = deep_merge({**server_args}, init_args)
            logger.info("Initialising experiment...")
            experiment = Experiment(merged_args)
            logger.info("Experiment initialised. Starting training thread.")
        except Exception as e:
            logger.exception(f"Failed to initialise experiment: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    experiment_thread = threading.Thread(
        target=experiment.run, daemon=True, name="ExperimentThread"
    )
    experiment_thread.start()
    logger.info("Experiment thread started.")
    return {"status": "started"}


@app.post("/stop")
async def stop():
    """Signals the running experiment to stop gracefully."""
    with experiment_lock:
        if experiment is None:
            return {"status": "nothing to stop"}
        experiment.stop()
    logger.info("Stop signal sent to experiment.")
    return {"status": "stopping"}


def run_server(args: dict):
    global server_args
    server_args = args
    setup_logging(args.get("logging", {}).get("level", "INFO"))
    port = int(os.environ.get("SERVER_PORT", 7777))
    logger.info(f"Starting ACID_sbs server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


@hydra.main(config_path="../config", config_name="default", version_base="1.2")
def main(cfg: DictConfig) -> None:
    if cfg.override != "":
        try:
            config_overrides = OmegaConf.load(
                hydra.utils.get_original_cwd() + f"/config/overrides/{cfg.override}.yaml"
            )
            cfg = OmegaConf.merge(cfg, config_overrides)
        except Exception as e:
            print(f"[ERROR] Failed to load override '{cfg.override}': {e}")
            raise
    args = OmegaConf.to_container(cfg, resolve=True)
    run_server(args)


if __name__ == "__main__":
    main()
