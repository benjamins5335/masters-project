pip3 install -r ../requirements.txt
python3 scripts/download_imagenet.py
python3 scripts/generate_images.py
python3 scripts/preprocess.py