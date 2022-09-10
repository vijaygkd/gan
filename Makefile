ROOT_DIR = ${PWD}

export PYTHONPATH=$(ROOT_DIR)

jupyter:
	pipenv run jupyter lab

train:
	pipenv run python dqn/train.py
