.PHONY: verify test check

verify:
	python3 verify_submission.py

test:
	python3 -m unittest discover -s tests -v

check: test verify
	python3 -m py_compile *.py recon/*.py
