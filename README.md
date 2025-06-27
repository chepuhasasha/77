python3.10 -m venv venv

venv\Scripts\activate 

python -m pip install --upgrade pip

pip install -r requirements.txt

pip install torch==2.3.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
