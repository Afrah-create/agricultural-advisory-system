"""
GitHub Model Loader for Agricultural Advisory System

This module provides functionality to load models from GitHub repositories
and integrate them with the Streamlit application.
"""

import os
import json
import requests
import pickle
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubModelLoader:
    """
    Load models from GitHub repositories for the agricultural advisory system
    """
    
    def __init__(self, github_repo: str, branch: str = "main", token: Optional[str] = None):
        """
        Initialize the GitHub model loader
        
        Args:
            github_repo: GitHub repository in format "owner/repo"
            branch: Branch name (default: "main")
            token: GitHub personal access token (optional, for private repos)
        """
        self.github_repo = github_repo
        self.branch = branch
        self.token = token
        self.base_url = f"https://api.github.com/repos/{github_repo}"
        self.raw_base_url = f"https://raw.githubusercontent.com/{github_repo}/{branch}"
        
        # Model cache directory
        self.cache_dir = Path("model_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized GitHub model loader for {github_repo}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests"""
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in the GitHub repository
        
        Returns:
            List of model information dictionaries
        """
        try:
            # Get repository contents from root directory (since models are in root)
            url = f"{self.base_url}/contents"
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            
            models = []
            for item in response.json():
                if item["type"] == "file" and item["name"].endswith((".json", ".pkl", ".joblib")):
                    models.append({
                        "name": item["name"],
                        "path": item["path"],
                        "size": item["size"],
                        "download_url": item["download_url"]
                    })
            
            # Also check if there's a models directory
            try:
                models_url = f"{self.base_url}/contents/models"
                models_response = requests.get(models_url, headers=self._get_headers())
                if models_response.status_code == 200:
                    for item in models_response.json():
                        if item["type"] == "file" and item["name"].endswith((".json", ".pkl", ".joblib")):
                            models.append({
                                "name": item["name"],
                                "path": item["path"],
                                "size": item["size"],
                                "download_url": item["download_url"]
                            })
            except:
                pass  # models directory doesn't exist, that's fine
            
            logger.info(f"Found {len(models)} models in repository")
            return models
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def download_model(self, model_name: str, force_download: bool = False) -> Optional[str]:
        """
        Download a model from GitHub
        
        Args:
            model_name: Name of the model file
            force_download: Force download even if cached
            
        Returns:
            Local path to the downloaded model, or None if failed
        """
        cache_path = self.cache_dir / model_name
        
        # Check if model is already cached
        if cache_path.exists() and not force_download:
            logger.info(f"Using cached model: {model_name}")
            return str(cache_path)
        
        try:
            # Try downloading from root directory first (for PKL files)
            url = f"{self.raw_base_url}/{model_name}"
            response = requests.get(url, headers=self._get_headers())
            
            # If not found in root, try models directory
            if response.status_code == 404:
                url = f"{self.raw_base_url}/models/{model_name}"
                response = requests.get(url, headers=self._get_headers())
            
            response.raise_for_status()
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded model: {model_name}")
            return str(cache_path)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return None
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load a model from GitHub
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Loaded model object, or None if failed
        """
        model_path = self.download_model(model_name)
        if not model_path:
            return None
        
        try:
            if model_name.endswith('.json'):
                with open(model_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif model_name.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            else:
                logger.error(f"Unsupported model format: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def load_knowledge_graph(self) -> Optional[Dict[str, Any]]:
        """
        Load the knowledge graph from GitHub
        
        Returns:
            Knowledge graph dictionary, or None if failed
        """
        return self.load_model("knowledge_graph.json")
    
    def load_embeddings(self, embedding_type: str = "graphsage") -> Optional[Dict[str, Any]]:
        """
        Load embeddings from GitHub
        
        Args:
            embedding_type: Type of embeddings to load (graphsage, gcn, transe, etc.)
            
        Returns:
            Embeddings dictionary, or None if failed
        """
        model_name = f"{embedding_type}_embeddings.json"
        return self.load_model(model_name)
    
    def load_rule_engine(self) -> Optional[Dict[str, Any]]:
        """
        Load the rule engine configuration from GitHub
        
        Returns:
            Rule engine configuration, or None if failed
        """
        return self.load_model("rule_engine_config.json")
    
    def load_crop_database(self) -> Optional[Dict[str, Any]]:
        """
        Load the crop database from GitHub
        
        Returns:
            Crop database dictionary, or None if failed
        """
        return self.load_model("crop_database.json")
    
    def load_cropping_planner(self) -> Optional[Any]:
        """
        Load the cropping planner model from GitHub
        
        Returns:
            Cropping planner model, or None if failed
        """
        return self.load_model("cropping_planner.pkl")
    
    def load_integrated_advisor(self) -> Optional[Any]:
        """
        Load the integrated advisor model from GitHub
        
        Returns:
            Integrated advisor model, or None if failed
        """
        return self.load_model("integrated_advisor.pkl")
    
    def load_rule_engine(self) -> Optional[Any]:
        """
        Load the rule engine model from GitHub
        
        Returns:
            Rule engine model, or None if failed
        """
        return self.load_model("rule_engine.pkl")
    
    def load_uncertainty_calibrator(self) -> Optional[Any]:
        """
        Load the uncertainty calibrator model from GitHub
        
        Returns:
            Uncertainty calibrator model, or None if failed
        """
        return self.load_model("uncertainty_calibrator.pkl")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Model information dictionary, or None if failed
        """
        try:
            url = f"{self.base_url}/contents/models/{model_name}"
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None
    
    def clear_cache(self):
        """Clear the model cache"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            logger.info("Model cache cleared")
    
    def get_cache_size(self) -> int:
        """Get the size of the model cache in bytes"""
        total_size = 0
        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size


class ModelManager:
    """
    Manager class for handling multiple model types and GitHub integration
    """
    
    def __init__(self, github_repo: str, branch: str = "main", token: Optional[str] = None):
        """
        Initialize the model manager
        
        Args:
            github_repo: GitHub repository in format "owner/repo"
            branch: Branch name (default: "main")
            token: GitHub personal access token (optional, for private repos)
        """
        self.loader = GitHubModelLoader(github_repo, branch, token)
        self.models = {}
        self.model_info = {}
        
        logger.info(f"Initialized ModelManager for {github_repo}")
    
    def load_all_models(self) -> Dict[str, Any]:
        """
        Load all available models from GitHub
        
        Returns:
            Dictionary of loaded models
        """
        available_models = self.loader.list_available_models()
        
        for model_info in available_models:
            model_name = model_info["name"]
            logger.info(f"Loading model: {model_name}")
            
            model = self.loader.load_model(model_name)
            if model is not None:
                self.models[model_name] = model
                self.model_info[model_name] = model_info
                logger.info(f"Successfully loaded: {model_name}")
            else:
                logger.warning(f"Failed to load: {model_name}")
        
        logger.info(f"Loaded {len(self.models)} models successfully")
        return self.models
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """
        Get a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model object, or None if not found
        """
        return self.models.get(model_name)
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all models
        
        Returns:
            Dictionary with model status information
        """
        status = {}
        for model_name, model in self.models.items():
            status[model_name] = {
                "loaded": True,
                "size": len(str(model)) if model else 0,
                "info": self.model_info.get(model_name, {})
            }
        
        return status
    
    def refresh_models(self) -> Dict[str, Any]:
        """
        Refresh all models from GitHub
        
        Returns:
            Dictionary of refreshed models
        """
        logger.info("Refreshing all models from GitHub...")
        self.loader.clear_cache()
        return self.load_all_models()


# Example usage and configuration
def create_model_manager_from_config(config_file: str = "github_config.json") -> ModelManager:
    """
    Create a ModelManager from configuration file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        ModelManager instance
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return ModelManager(
            github_repo=config["github_repo"],
            branch=config.get("branch", "main"),
            token=config.get("token")
        )
    except FileNotFoundError:
        logger.warning(f"Configuration file {config_file} not found, using defaults")
        return ModelManager("your-username/your-repo")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return ModelManager("your-username/your-repo")


if __name__ == "__main__":
    # Example usage
    print("GitHub Model Loader Example")
    print("=" * 40)
    
    # Initialize model manager
    manager = ModelManager("your-username/crop-recommendation-models")
    
    # List available models
    available_models = manager.loader.list_available_models()
    print(f"Available models: {len(available_models)}")
    for model in available_models:
        print(f"  - {model['name']}")
    
    # Load all models
    models = manager.load_all_models()
    print(f"Loaded models: {len(models)}")
    
    # Get model status
    status = manager.get_model_status()
    for model_name, info in status.items():
        print(f"  - {model_name}: {'✅' if info['loaded'] else '❌'}")
