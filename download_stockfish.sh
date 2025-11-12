"""
Downloads and extracts stockfish and installs tinker cookbook
"""

wget https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar
tar xvf stockfish-ubuntu-x86-64-avx2.tar

git clone https://github.com/thinking-machines-lab/tinker-cookbook
cd tinker-cookbook
pip install -e .
