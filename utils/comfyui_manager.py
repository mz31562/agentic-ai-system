# utils/comfyui_manager.py
import subprocess
import sys
import os
import time
import requests
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ComfyUIManager:
    """Manages ComfyUI server lifecycle as a subprocess"""
    
    def __init__(
        self,
        comfyui_path: Optional[str] = None,
        conda_env: str = "comfyui",
        host: str = "127.0.0.1",
        port: int = 8188
    ):
        """
        Initialize ComfyUI Manager
        
        Args:
            comfyui_path: Path to ComfyUI directory
            conda_env: Name of conda environment for ComfyUI
            host: Host to bind ComfyUI to
            port: Port to run ComfyUI on
        """
        # Find ComfyUI path
        if comfyui_path is None:
            possible_paths = [
                Path("ComfyUI"),
                Path("../ComfyUI"),
                Path.home() / "ComfyUI",
                Path("C:/Users/MohammedZaid/Desktop/agentic-ai-system/ComfyUI")
            ]
            
            for path in possible_paths:
                if path.exists() and (path / "main.py").exists():
                    self.comfyui_path = path
                    break
            else:
                raise FileNotFoundError(
                    "ComfyUI not found. Please set COMFYUI_PATH in .env or "
                    "place ComfyUI in one of: " + ", ".join(str(p) for p in possible_paths)
                )
        else:
            self.comfyui_path = Path(comfyui_path)
        
        self.conda_env = conda_env
        self.host = host
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.url = f"http://{host}:{port}"
        
        logger.info(f"ComfyUI Manager initialized")
        logger.info(f"  Path: {self.comfyui_path}")
        logger.info(f"  Conda env: {conda_env}")
        logger.info(f"  URL: {self.url}")
    
    
    def is_running(self) -> bool:
        """Check if ComfyUI is already running"""
        try:
            response = requests.get(f"{self.url}/system_stats", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    
    def start(self, wait_for_ready: bool = True, timeout: int = 60) -> bool:
        """
        Start ComfyUI server as subprocess
        
        Args:
            wait_for_ready: Wait for server to be ready before returning
            timeout: Maximum seconds to wait for server startup
            
        Returns:
            True if started successfully, False otherwise
        """
        # Check if already running
        if self.is_running():
            logger.info("✅ ComfyUI is already running")
            return True
        
        logger.info("🚀 Starting ComfyUI server...")
        
        try:
            # Build command based on OS
            if sys.platform == "win32":
                # Windows: Use conda run
                cmd = [
                    "conda", "run", "-n", self.conda_env, "--no-capture-output",
                    "python", "main.py",
                    "--listen", self.host,
                    "--port", str(self.port)
                ]
            else:
                # Linux/Mac: Activate conda and run
                cmd = [
                    "bash", "-c",
                    f"source $(conda info --base)/etc/profile.d/conda.sh && "
                    f"conda activate {self.conda_env} && "
                    f"python main.py --listen {self.host} --port {self.port}"
                ]
            
            # Start ComfyUI process
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.comfyui_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            logger.info(f"   Process started with PID: {self.process.pid}")
            
            # Wait for server to be ready
            if wait_for_ready:
                logger.info(f"   Waiting for server to be ready (timeout: {timeout}s)...")
                
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self.is_running():
                        logger.info("✅ ComfyUI server is ready!")
                        return True
                    
                    # Check if process died
                    if self.process.poll() is not None:
                        stderr = self.process.stderr.read() if self.process.stderr else ""
                        logger.error(f"❌ ComfyUI process died during startup")
                        logger.error(f"   Error output: {stderr}")
                        return False
                    
                    time.sleep(1)
                
                logger.error(f"❌ ComfyUI server failed to start within {timeout}s")
                return False
            
            return True
            
        except FileNotFoundError:
            logger.error("❌ Conda not found. Is Anaconda/Miniconda installed?")
            logger.error("   Make sure 'conda' is in your PATH")
            return False
        
        except Exception as e:
            logger.error(f"❌ Failed to start ComfyUI: {e}", exc_info=True)
            return False
    
    
    def stop(self, timeout: int = 10):
        """
        Stop ComfyUI server gracefully
        
        Args:
            timeout: Seconds to wait before force kill
        """
        if self.process is None:
            logger.info("ComfyUI process not managed by this instance")
            return
        
        if self.process.poll() is not None:
            logger.info("ComfyUI process already stopped")
            return
        
        logger.info("🛑 Stopping ComfyUI server...")
        
        try:
            # Try graceful termination
            self.process.terminate()
            
            # Wait for process to end
            try:
                self.process.wait(timeout=timeout)
                logger.info("✅ ComfyUI stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if still running
                logger.warning("⚠️  Forcing ComfyUI to stop...")
                self.process.kill()
                self.process.wait()
                logger.info("✅ ComfyUI stopped (forced)")
        
        except Exception as e:
            logger.error(f"Error stopping ComfyUI: {e}")
    
    
    def get_status(self) -> dict:
        """Get ComfyUI server status"""
        return {
            "running": self.is_running(),
            "url": self.url,
            "path": str(self.comfyui_path),
            "conda_env": self.conda_env,
            "process_active": self.process is not None and self.process.poll() is None
        }