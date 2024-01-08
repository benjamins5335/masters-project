if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_name>"
    exit
fi

python3 evaluate.py --model_path $1 --whole_set --subclasses