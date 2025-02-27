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

**3. Torch:**
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

**4. Install the required packages for the model you are using:**
Check their github page for the required packages.

**5. Run the main file:**
```
python main.py
```

**6. To add a new model:**
```
Edit the main.py file and add the model to "choose_model" function.
```