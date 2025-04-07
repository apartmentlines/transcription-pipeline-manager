# Transcription Pipeline Manager

Orchestrates a transcription pipeline running on [RunPod](https://www.runpod.io). It operates on an hourly cycle: starts a configured [RunPod](https://www.runpod.io) pod, waits for it to become idle, triggers a remote processing job via HTTP POST, and manages status updates via a built-in local REST interface.

## Prerequisites

* [Python](https://www.python.org) 3.10+
* RunPod Account and API Key (required for interacting with RunPod services via `runpod-singleton`).
* A configured RunPod pod template/endpoint that exposes the following API:
  *   `GET /status`: Returns `{"status": "idle"}` when ready.
  *   `POST /run`: Accepts a JSON payload to trigger the pipeline.

## Installation

Install the package (preferably in a virtual environment):

```bash
pip install .
```

## Configuration

### RunPod API Key

The underlying `runpod-singleton` library requires your RunPod API key to interact with RunPod services (start, terminate pods). You **must** export this key to your environment:

```bash
export RUNPOD_API_KEY="your_actual_runpod_api_key"
```

### Manager Configuration

The manager itself requires configuration for its own operation (e.g., authenticating callbacks, constructing URLs). These are separate from the RunPod API key. Provide them via:

1.  **Environment Variables (Recommended):**
    *   `TRANSCRIPTION_API_KEY`: An API key used by the manager and its REST interface for securing callbacks.
    *   `TRANSCRIPTION_DOMAIN`: The base domain used for constructing callback URLs (e.g., `yourdomain.com`).
2.  **CLI Arguments:**
    *   `--api-key <manager-api-key>`
    *   `--domain <your-domain.com>`

*Note: CLI arguments override environment variables.*

### RunPod Pod Configuration

The manager uses `runpod-singleton` to manage the target pod. You **must** create a configuration file at:

```
runpod-singleton/transcription-pipeline.config.yaml
```

This file defines the pod template, GPU type, environment variables, etc., that `runpod-singleton` will use to start/manage the pod. Refer to the [runpod-singleton](https://github.com/apartmentlines/runpod-singleton) documentation for the required format.

### Other Manager Options

The `transcription-pipeline-manager` script accepts additional arguments for controlling limits, debugging, etc. Refer to the help message for details:

```bash
transcription-pipeline-manager --help
```

## Usage

### Running the Manager

This script starts the main orchestration loop and the built-in REST interface for callbacks.

```bash
# Ensure RUNPOD_API_KEY is exported first!
export RUNPOD_API_KEY="your_actual_runpod_api_key"

# Using environment variables for manager config
export TRANSCRIPTION_API_KEY="your_manager_api_key"
export TRANSCRIPTION_DOMAIN="yourdomain.com"
transcription-pipeline-manager [OPTIONS]

# Or using CLI arguments for manager config
transcription-pipeline-manager --api-key "your_manager_api_key" --domain "yourdomain.com" [OPTIONS]
```

The manager will run indefinitely until interrupted (Ctrl+C). It automatically starts a local REST server (default: `http://127.0.0.1:8080`) to receive callbacks from the triggered pod. The manager constructs the `callback_url` as:

```python
return f"https://www.{domain}/api/transcription/logs?api_key={api_key}"
```

The callback URL can be used to send logs to a logging server, or back to the local REST server if it has an internet accessible address.

### Standalone REST Interface (for Testing)

While the manager runs its own REST interface internally, a standalone script is provided primarily for testing or development purposes. If you need to run the callback receiver separately:

```bash
# Run locally on default host/port
rest-interface [--debug]

# Run with API key protection (key should match manager's TRANSCRIPTION_API_KEY)
rest-interface --api-key "your_manager_api_key" [--debug]
```

*Note: Running the `rest-interface` script separately is generally **not** required for normal operation of the `transcription-pipeline-manager`.*
