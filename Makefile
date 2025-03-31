# Instala las dependencias del proyecto
install:
	poetry install

# Ejecuta el script de ingestión de datos
ingest:
	poetry run python src/itemclassifier/stage_01_data_ingestion.py

# Ejecuta el script de entrenamiento del modelo
train:
	poetry run python src/itemclassifier/stage_02_model_training.py

evaluate:
	poetry run python src/itemclassifier/stage_03_model_evaluation.py

# Ejecuta los tests del proyecto con cobertura
test:
	poetry run pytest -vv --cov=src --cov-report=term-missing

# Formatea el código usando Black
format:
	poetry run black src/**/*.py tests/*.py

# Analiza el código en busca de errores estáticos con Pylint
lint:
	poetry run pylint --disable=R,C --extension-pkg-whitelist='pydantic' src/**/*.py tests/*.py --exit-zero

# Ejecuta formateo y análisis estático
refactor: format lint

# Ejecuta todos los pasos del flujo de trabajo
all: install lint test format

# Ayuda para mostrar el uso del Makefile
help:
	@echo "Comandos disponibles:"
	@echo "  make install   - Instala las dependencias del proyecto"
	@echo "  make ingest    - Ejecuta el script de ingestión de datos"
	@echo "  make train     - Ejecuta el script de entrenamiento del modelo"
	@echo "  make evaluate  - Ejecuta el script de evaluación del modelo"
	@echo "  make test      - Ejecuta los tests del proyecto con cobertura"
	@echo "  make format    - Formatea el código usando Black"
	@echo "  make lint      - Analiza el código en busca de errores estáticos con Pylint"
	@echo "  make refactor  - Ejecuta formateo y análisis estático"
	@echo "  make all       - Ejecuta todos los pasos del flujo de trabajo"
