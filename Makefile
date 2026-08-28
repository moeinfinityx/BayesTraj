.PHONY: reproduce verify test clean

reproduce:
	python scripts/reproduce.py

verify:
	python scripts/verify.py

test:
	pytest -q

clean:
	find outputs -type f ! -name .gitkeep -delete

