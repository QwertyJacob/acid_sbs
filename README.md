## acid_sbs

Prototypical learning experiment server on synthetic data.

### Run

```bash
# Default config
python -m src.server

# With an override file (config/overrides/my_exp.yaml)
python -m src.server override=my_exp

# Custom port
SERVER_PORT=8888 python -m src.server
```

### API

| Method | Endpoint | Body            | Action                                                                                      |
|--------|----------|-----------------|---------------------------------------------------------------------------------------------|
| GET    | /        | —               | Health check                                                                                |
| POST   | /start   | dict (optional) | Initialise **and** start experiment. Body is deep-merged on top of YAML config at runtime. |
| POST   | /stop    | —               | Signal graceful stop                                                                        |

### Quick test

```bash
# Start with default config
curl -X POST http://localhost:7777/start -H "Content-Type: application/json" -d '{}'

# Start with runtime overrides (e.g. change run name and disable plots)
curl -X POST http://localhost:7777/start \
  -H "Content-Type: application/json" \
  -d '{"wandb": {"wb_run_name": "my_test_run"}, "prototypical": {"train_episodes": 50}}'

# Watch logs in the terminal, then stop:
curl -X POST http://localhost:7777/stop
```
