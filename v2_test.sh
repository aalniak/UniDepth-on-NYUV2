# Initialize Conda
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path if needed

conda activate Unidepth # Replace with your environment name you created (and installed requirements) for UniDepth
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

python test_nyu.py --model v2
