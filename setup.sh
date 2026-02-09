set -e 

echo "===== Multilingual Ticket Routing Setup ====="

if [ ! -d "venv" ]; then 
    echo "Creating Python virtual environment..."
    python3 -m venv venv 
fi 

source venv/bin/activate 

pip install --upgrade pip 

echo "Installing Python dependencies..."
pip install -r requirements.txt 

echo "Setting up folders..."
for folder in data/raw data/processed experiments logs checkpoints; do
    if [ ! -d "$folder" ]; then
        mkdir -p "$folder"
        echo "Created folder: $folder"
    fi
done


if python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then 
    echo "GPU detected: Pytorch for CUDA." 
else 
    echo "WARNING: No GPU detected. Training will run on CPU." 
fi 

if [ "$1" == "docker" ]; then 
    echo "Building Docker image..." 
    docker build -t multilingual-ticket-routing:latest . 
fi 

echo "==== Setup Complete ===="
echo "Activate environment with: source venv/bin/activate"
echo "Run preprocessing: python src/preprocessing/preprocess.py --input data/raw --output data/processed"
echo "Run training: torchrun --nproc_per_node=NUM_GPUS src/training/train_ddp.py"