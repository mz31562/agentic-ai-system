# test_comfyui_connection.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_comfyui():
    """Test ComfyUI connection"""
    print("\n" + "="*70)
    print("🔍 COMFYUI CONNECTION TEST")
    print("="*70)
    
    # Check environment variables
    print("\n1️⃣ Environment Variables:")
    print(f"   IMAGE_BACKEND: {os.getenv('IMAGE_BACKEND', 'NOT SET')}")
    print(f"   IMAGE_MODEL: {os.getenv('IMAGE_MODEL', 'NOT SET')}")
    print(f"   COMFYUI_URL: {os.getenv('COMFYUI_URL', 'NOT SET')}")
    
    # Test connection
    url = os.getenv('COMFYUI_URL', 'http://127.0.0.1:8188')
    print(f"\n2️⃣ Testing connection to: {url}")
    
    try:
        response = requests.get(f"{url}/system_stats", timeout=5)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ SUCCESS - ComfyUI is accessible!")
            stats = response.json()
            print(f"   System Stats: {stats}")
        else:
            print(f"   ❌ FAILED - Status: {response.status_code}")
            print(f"   Response: {response.text}")
    
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ CONNECTION FAILED")
        print(f"   Error: {e}")
        print(f"\n   Make sure ComfyUI is running:")
        print(f"   python main.py --listen --port 8188")
    
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    test_comfyui()