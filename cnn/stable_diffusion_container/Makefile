PROJECT_NAME = gen-sd-image
CLASSES = cat car

all: venv docker_run_multiple

venv:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

docker_run:
	docker run \
	--rm --gpus all \
	--name bs214-cs5199 \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-v /home/bs214/cs5199/cnn/stable_diffusion_container/data:/app/data \
	-it $(PROJECT_NAME) cat

docker_run_multiple:
	@for class in $(CLASSES); do echo "Starting container for $$class..."; \
        docker run --rm -d --gpus all \
        --name "bs214-cs5199_$$class" \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v /home/bs214/cs5199/cnn/stable_diffusion_container/data:/app/data \
        $(PROJECT_NAME) $$class; \
    done

.PHONY: all venv docker_run
