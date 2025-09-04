.PHONY: run api test

run:
	uvicorn app.main:app --reload

api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q
