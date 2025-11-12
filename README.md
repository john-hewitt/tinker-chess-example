# Setup

Set up your favorite virtual environment and then:
```
git clone https://github.com/thinking-machines-lab/tinker-cookbook
cd tinker-cookbook
pip install -e .
```
And next download stockfish:
```
bash download_stockfish.sh
```

## Set your tinker API key
However you decide to do that, e.g., 

```
export TINKER_API_KEY=...
```

## Set up the chess dataset

```
python prep_uci_dataset.py
```

# Run an RL training run

Run

```
python -m rl_chess_loop
```
