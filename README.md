**1. Create Virtual Envoironment:** 
```
python -m venv venv 
.\venv\Scripts\activate
```
**2. Set up your Qualcomm AI Hub development environment:** 
```
pip install qai-hub
qai-hub configure --api_token <API_TOKEN>
```
**3. Fetch device list:**
```
qai-hub list-devices
```
Device to choose: "QCS8550 (Proxy)"

**4. Torch:**
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```


