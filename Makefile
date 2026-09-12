install:
	pip install -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check .

gradient-check:
	python scripts/gradient_check.py

xor:
	python examples/xor.py

mnist:
	python examples/mnist_mlp.py --epochs 10
