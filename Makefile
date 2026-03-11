# Makefile for Vendor Invoice Intelligence
# Simplifies the development workflow

.PHONY: setup data train evaluate app docker-build docker-run

# Initial Environment Setup
setup:
	pip install -r requirements.txt
	pip install -e .

# Data Generation and SQL Transformation
data:
	python src/generate_data.py
	python src/database_manager.py

# Model Training
train:
	python src/train_regression_model.py
	python src/train_classification_model.py

# Model Evaluation and Interpretability
evaluate:
	python src/evaluate_model.py
	python src/model_explainability.py

# Launch Streamlit App
app:
	streamlit run app/streamlit_app.py

# Docker Tasks
docker-build:
	docker build -t vendor-invoice-ai .

docker-run:
	docker run -p 8501:8501 vendor-invoice-ai
