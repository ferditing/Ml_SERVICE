Rebuild & test model

1) Create venv and install
powershell:
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

2) Train model (writes decision_tree_model.pkl)
python train_decision_tree.py

3) Sanity test
python test_predict.py

4) Run API
uvicorn ml_service:app --reload --port 8000
# ml_service.py will load the saved decision_tree_model.pkl by default.