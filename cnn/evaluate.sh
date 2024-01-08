if [ "$#" -ne 2 ]; then
    echo "Usage: $0 --model_path <model_name>"
    exit
fi

python3 evaluate.py --model_path $2 --whole_set --subclasses