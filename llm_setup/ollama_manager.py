#llm_setup/ollama_manager.py
import logging
import os
import mlflow
from ollama import Client
from llm_setup.config import LLMConfig

# Get the host from the environment variable set in docker-compose
# Defaults to host.docker.internal if not set
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

def get_client():
    """Helper to get a configured Ollama client."""
    return Client(host=OLLAMA_HOST)

def setup_ollama(config: LLMConfig) -> bool:
    """
    Connect to the remote Ollama service (on Windows) and ensure the model exists.
    """
    logging.info(f"Connecting to Ollama at {OLLAMA_HOST} for model: {config.model_name}")
    mlflow.log_param("llm_model", config.model_name)
    mlflow.log_param("ollama_host", OLLAMA_HOST)

    client = get_client()

    try:

        response = client.list()
        logging.info("Successfully connected to Ollama service on host.")
        

        existing_models = [m.get('model') for m in response.get('models', [])]
        

        model_exists = any(config.model_name in m for m in existing_models)

        if not model_exists:
            logging.info(f"Model {config.model_name} not found. Triggering pull via API...")
            

            progress = client.pull(config.model_name, stream=False)
            
            logging.info(f"Model {config.model_name} pulled successfully.")
            mlflow.log_param("model_pulled", "true")
        else:
            logging.info(f"Model {config.model_name} is already available.")
            
        return True

    except Exception as e:
        logging.error(f"Failed to communicate with Ollama at {OLLAMA_HOST}.")
        logging.error(f"Error details: {str(e)}")
        logging.warning("PLEASE ENSURE: 1. Ollama is running on Windows. 2. You ran 'set OLLAMA_HOST=0.0.0.0' in Windows PowerShell.")
        return False

def query_llm(prompt: str, config: LLMConfig) -> str | None:
    """Query the LLM with a given prompt using the remote client."""
    try:
        client = get_client()
        
        response = client.chat(
            model=config.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
        
    except Exception as e:
        logging.exception("LLM query failed: %s", e)
        mlflow.log_param("llm_error", str(e))
        return None