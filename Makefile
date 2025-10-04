.PHONY: install run prod fmt lint clean

install:
	@test -d .venv || python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run:
	. .venv/bin/activate && uvicorn app.main:app --reload

prod:
	. .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2

fmt:
	. .venv/bin/activate && python -m black app

lint:
	. .venv/bin/activate && python -m flake8 app

clean:
	rm -rf .venv .cache __pycache__ */__pycache__ outputs/*.json