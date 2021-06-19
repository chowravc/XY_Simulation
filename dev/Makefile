.PHONY: clean-dirs clean-pyc

clean-dirs: 
	#delete all empty directories
	find ./ -empty -type d -delete

clean-pyc: 
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} + 
	find . -name '*~*' -exec rm --force {} +
	
