import os
from pathlib import Path
import requests
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ComfyUI model downloads and verification"""
    
    MODELS = {
        "sd15": {
            "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
            "filename": "v1-5-pruned-emaonly.safetensors",
            "size_mb": 4000,
            "description": "Stable Diffusion 1.5 (Faster, less VRAM)"
        },
        "sdxl": {
            "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
            "filename": "sd_xl_base_1.0.safetensors",
            "size_mb": 6900,
            "description": "Stable Diffusion XL (Higher quality)"
        }
    }
    
    def __init__(self, comfyui_path: str = None):
        """
        Initialize Model Manager
        
        Args:
            comfyui_path: Path to ComfyUI directory (auto-detects if None)
        """
        if comfyui_path is None:
            # Try to find ComfyUI directory
            possible_paths = [
                Path("ComfyUI"),
                Path("../ComfyUI"),
                Path.home() / "ComfyUI",
                Path("C:/Users/MohammedZaid/Desktop/agentic-ai-system/ComfyUI")
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.comfyui_path = path
                    break
            else:
                # Default to first option
                self.comfyui_path = possible_paths[0]
        else:
            self.comfyui_path = Path(comfyui_path)
        
        self.checkpoint_dir = self.comfyui_path / "models" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model directory: {self.checkpoint_dir.absolute()}")
    
    
    def check_model_exists(self, model_name: str) -> bool:
        """
        Check if a model file exists
        
        Args:
            model_name: Name of model (sd15 or sdxl)
            
        Returns:
            True if model exists, False otherwise
        """
        if model_name not in self.MODELS:
            logger.warning(f"Unknown model name: {model_name}")
            return False
        
        model_info = self.MODELS[model_name]
        model_path = self.checkpoint_dir / model_info["filename"]
        
        exists = model_path.exists()
        if exists:
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Model found: {model_path.name} ({size_mb:.0f}MB)")
        else:
            logger.warning(f"❌ Model not found: {model_path.name}")
        
        return exists
    
    
    def download_model(self, model_name: str, force: bool = False) -> str:
        """
        Download a model file
        
        Args:
            model_name: Name of model to download (sd15 or sdxl)
            force: Force re-download even if file exists
            
        Returns:
            Path to downloaded model file
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(self.MODELS.keys())}")
        
        model_info = self.MODELS[model_name]
        model_path = self.checkpoint_dir / model_info["filename"]
        
        # Check if already exists
        if model_path.exists() and not force:
            logger.info(f"✅ Model already exists: {model_path.name}")
            return str(model_path)
        
        # Show download info
        print("\n" + "="*70)
        print(f"📥 DOWNLOADING: {model_info['description']}")
        print("="*70)
        print(f"File: {model_info['filename']}")
        print(f"Size: ~{model_info['size_mb']}MB ({model_info['size_mb']/1024:.1f}GB)")
        print(f"Destination: {model_path}")
        print(f"URL: {model_info['url']}")
        print("="*70)
        print("⏳ This may take 10-30 minutes depending on your internet speed...")
        print("="*70 + "\n")
        
        # Download with progress bar
        try:
            response = requests.get(model_info["url"], stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=model_info["filename"],
                ncols=80
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print("\n" + "="*70)
            print(f"✅ DOWNLOAD COMPLETE: {model_path.name}")
            print("="*70 + "\n")
            
            logger.info(f"✅ Model downloaded successfully: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
                logger.info("Cleaned up partial download")
            
            raise
    
    
    def list_available_models(self) -> list:
        """
        List all models in checkpoint directory
        
        Returns:
            List of Path objects for found models
        """
        models = list(self.checkpoint_dir.glob("*.safetensors")) + \
                 list(self.checkpoint_dir.glob("*.ckpt"))
        
        if models:
            logger.info(f"📦 Found {len(models)} model(s) in {self.checkpoint_dir}:")
            for model in models:
                size_mb = model.stat().st_size / (1024 * 1024)
                logger.info(f"   - {model.name} ({size_mb:.0f}MB)")
        else:
            logger.warning(f"⚠️  No models found in {self.checkpoint_dir}")
        
        return models
    
    
    def get_recommended_model(self, prefer_quality: bool = True) -> str:
        """
        Get recommended model based on preferences
        
        Args:
            prefer_quality: If True, recommend SDXL; if False, recommend SD1.5
            
        Returns:
            Model name (sdxl or sd15)
        """
        return "sdxl" if prefer_quality else "sd15"
    
    
    def verify_and_download(self, model_name: str = "sdxl", interactive: bool = True) -> bool:
        """
        Verify model exists, offer to download if missing
        
        Args:
            model_name: Model to check (sdxl or sd15)
            interactive: If True, prompt user before downloading
            
        Returns:
            True if model is available (exists or downloaded), False otherwise
        """
        # Check if model exists
        if self.check_model_exists(model_name):
            return True
        
        # Model doesn't exist
        print("\n" + "="*70)
        print("⚠️  MODEL NOT FOUND")
        print("="*70)
        print(f"The {model_name.upper()} model is not installed.")
        print(f"ComfyUI needs this model to generate images.\n")
        
        model_info = self.MODELS.get(model_name, {})
        print(f"Model: {model_info.get('description', model_name)}")
        print(f"Size: ~{model_info.get('size_mb', 0)/1024:.1f}GB")
        print(f"Location: {self.checkpoint_dir}")
        print("="*70 + "\n")
        
        if interactive:
            # Ask user if they want to download
            response = input(f"📥 Download {model_name.upper()} model now? [y/N]: ").strip().lower()
            
            if response == 'y':
                try:
                    self.download_model(model_name)
                    return True
                except Exception as e:
                    print(f"\n❌ Download failed: {e}")
                    print("You can manually download from:")
                    print(f"   {model_info.get('url', 'N/A')}")
                    print(f"And place it in: {self.checkpoint_dir}\n")
                    return False
            else:
                print("\n⚠️  Skipping download. System will run in MOCK mode.")
                print("To download later, run:")
                print(f"   python -c 'from utils.model_manager import ModelManager; ModelManager().download_model(\"{model_name}\")'\n")
                return False
        else:
            # Non-interactive mode - just download
            try:
                self.download_model(model_name)
                return True
            except:
                return False


# CLI interface for direct usage
if __name__ == "__main__":
    import sys
    
    manager = ModelManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "list":
            manager.list_available_models()
        
        elif command == "download":
            model = sys.argv[2] if len(sys.argv) > 2 else "sdxl"
            manager.download_model(model)
        
        elif command == "check":
            model = sys.argv[2] if len(sys.argv) > 2 else "sdxl"
            exists = manager.check_model_exists(model)
            print(f"\n{'✅' if exists else '❌'} Model {model}: {'Found' if exists else 'Not found'}\n")
        
        else:
            print("Usage:")
            print("  python model_manager.py list              - List installed models")
            print("  python model_manager.py check [sdxl|sd15] - Check if model exists")
            print("  python model_manager.py download [sdxl|sd15] - Download model")
    
    else:
        print("\n🔍 Checking for models...\n")
        manager.list_available_models()
        
        if not manager.check_model_exists("sdxl") and not manager.check_model_exists("sd15"):
            print("\n💡 No models found. Download with:")
            print("   python model_manager.py download sdxl")
            print("   python model_manager.py download sd15")