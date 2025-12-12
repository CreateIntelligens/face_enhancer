# Face Enhancer Service

Minimal FastAPI microservice that wraps the FaceFusion `face_enhancer` processor and its face selector logic. A trimmed FaceFusion source tree is bundled in this repo; models download on first use into `.assets`. You can still override with `FACEFUSION_PATH` if you want to point at another checkout.

## Quick start

1. Install dependencies in a virtualenv:
   ```bash
   cd face_enhancer
   pip install -r requirements.txt
   pip install -e .
   ```
2. Run the API:
   ```bash
   uvicorn face_enhancer.app:app --reload --port 8005
   ```
   Open `http://localhost:8005/` for the demo UI.
3. Call the service:
   ```bash
   curl -X POST http://localhost:8005/enhance \
     -H "Content-Type: application/json" \
     -d @payload.json  # see /docs for schema
   ```

## Docker

Build and run from the parent directory that contains the `face_enhancer` folder:

```bash
docker build -f face_enhancer/Dockerfile -t face-enhancer-service .
docker run --rm -p 8005:8005 face-enhancer-service
```

### docker-compose

From the same parent directory (contains `face_enhancer`):

```bash
docker compose -f face_enhancer/docker-compose.yml up --build
```

This maps port `8005:8005` and mounts `./.assets` and `./.caches` so model downloads persist across runs.

## Notes

- Defaults mirror FaceFusion CLI/UI (CPU execution, `gfpgan_1.4`, selector `reference` mode).
- Models download on demand into `.assets`/`.caches` within this project.
