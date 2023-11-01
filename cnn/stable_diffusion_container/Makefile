PROJECT_NAME = gen-sd-image

# 'all' target is the default when make is run without arguments.
all: venv docker_run

# Set up Python virtual environment and install dependencies.
venv:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

# Run Docker container
docker_run:
	docker run --rm --gpus all --name bs214-cs5199 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v "$(pwd)/data:/app/data" -it $(PROJECT_NAME) cat

# Other potential rules can be added here

.PHONY: all venv docker_run
