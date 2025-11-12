#!/usr/bin/env python3
"""
Load Lichess tournament chess games dataset and create a dataset of move prefixes in UCI format.
"""

import chess
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import argparse
import re


def extract_uci_moves_from_movetext(movetext):
    """Extract UCI moves from annotated movetext format.
    
    Format example: "1. e4 { [%eval 0.24] [%clk 1:40:57] } 1... c5 ..."
    """
    try:
        # Remove all annotations in curly braces
        cleaned = re.sub(r'\{[^}]*\}', '', movetext)
        
        # Remove move numbers (like "1.", "1...", "2.", etc.)
        cleaned = re.sub(r'\d+\.\.?\.?\s*', '', cleaned)
        
        # Remove result at the end (like "0-1", "1-0", "1/2-1/2")
        cleaned = re.sub(r'\s*(0-1|1-0|1/2-1/2)\s*$', '', cleaned)
        
        # Split by whitespace and filter out empty strings
        move_tokens = [m.strip() for m in cleaned.split() if m.strip()]
        
        # Convert SAN moves to UCI format
        board = chess.Board()
        uci_moves = []
        
        for san_move in move_tokens:
            try:
                move = board.parse_san(san_move)
                uci_moves.append(move.uci())
                board.push(move)
            except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
                # Skip invalid moves
                break
        
        return uci_moves
    except Exception as e:
        # Skip games that can't be parsed
        return []


def get_board_ascii(uci_moves):
    """Get ASCII representation of board after applying UCI moves."""
    board = chess.Board()
    for uci_move in uci_moves:
        try:
            move = chess.Move.from_uci(uci_move)
            board.push(move)
        except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
            # Return empty board if move is invalid
            return str(chess.Board())
    return str(board)


def generate_move_prefixes(uci_moves, min_moves=15):
    """Generate all prefixes of a move list with their board states.
    
    Only includes prefixes with at least min_moves moves.
    """
    prefixes = []
    board_states = []
    
    # Only generate prefixes with at least min_moves moves
    start_idx = max(1, min_moves)
    for i in range(start_idx, len(uci_moves) + 1):
        prefix = uci_moves[:i]
        prefix_str = ' '.join(prefix)
        prefixes.append(prefix_str)
        
        # Get board state for this prefix
        board_ascii = get_board_ascii(prefix)
        board_states.append(board_ascii)
    
    return prefixes, board_states


def process_dataset(input_dataset, max_games=None):
    """Process the dataset to create move prefixes."""
    all_prefixes = []
    all_board_states = []
    
    num_games = len(input_dataset)
    if max_games:
        num_games = min(num_games, max_games)
    
    print(f"Processing {num_games} games...")
    
    for i in tqdm(range(num_games)):
        game = input_dataset[i]
        
        # Get movetext field
        if 'movetext' not in game:
            continue
        
        movetext = game['movetext']
        if movetext is None:
            continue
        
        # Extract UCI moves
        uci_moves = extract_uci_moves_from_movetext(movetext)
        
        if len(uci_moves) == 0:
            continue
        
        # Skip games with fewer than 15 moves (won't generate any prefixes)
        if len(uci_moves) < 15:
            continue
        
        # Generate prefixes and board states
        prefixes, board_states = generate_move_prefixes(uci_moves)
        all_prefixes.extend(prefixes)
        all_board_states.extend(board_states)
    
    print(f"Generated {len(all_prefixes)} move prefixes")
    dataset = Dataset.from_dict({
        'uci_prefix': all_prefixes,
        'board_ascii': all_board_states
    })
    return DatasetDict({'train': dataset})


def main():
    parser = argparse.ArgumentParser(description='Load Lichess dataset and create move prefixes')
    parser.add_argument('--max-games', type=int, default=None, 
                       help='Maximum number of games to process (for testing)')
    parser.add_argument('--output', type=str, default='lichess_move_prefixes',
                       help='Output dataset name/path')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use')
    
    args = parser.parse_args()
    
    print("Loading Lichess tournament chess games dataset...")
    dataset = load_dataset("Lichess/tournament-chess-games", split=args.split)
    
    print(f"Dataset loaded: {len(dataset)} games")
    
    # Process the dataset
    prefix_dataset = process_dataset(dataset, max_games=args.max_games)
    
    # Save the dataset
    print(f"Saving dataset to '{args.output}'...")
    prefix_dataset.save_to_disk(args.output)
    
    print("Done!")
    print(f"\nSample prefixes with board states:")
    train_split = prefix_dataset['train']
    for i in range(min(10, len(train_split))):
        print(f"\n  {i+1}. UCI: {train_split[i]['uci_prefix']}")
        print(f"     Board:")
        print(train_split[i]['board_ascii'])


if __name__ == '__main__':
    main()

