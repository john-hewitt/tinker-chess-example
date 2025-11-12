"""
John hacking on RL for chess move centipawn loss improvement via thinking traces.
"""
import logging
import subprocess
import re
import random
import chess

import time
from concurrent.futures import Future

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/rl-chess-loop-2"
    model_name: str = "openai/gpt-oss-20b"
    batch_size: int = 128
    group_size: int = 16
    learning_rate: float = 4e-5
    max_length: int = 32768
    lora_rank: int = 16
    save_every: int = 20
    #max_tokens: int = 256
    max_tokens: int = 1024


strfile = """
uci
isready
ucinewgame
position startpos moves {}
go movetime 500 searchmoves {}
"""

def call_stockfish(input_str):
  # Your input string (instead of writing to tmp)
  #input_str = "position startpos\n go depth 10\n"

  # Run stockfish as a subprocess
  proc = subprocess.Popen(
      ["./stockfish/stockfish-ubuntu-x86-64-avx2"],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True
  )

  # Send your commands
  proc.stdin.write(input_str)
  proc.stdin.flush()

  # <-- manual wait (like `sleep 1`)
  time.sleep(0.51)

  # Tell Stockfish to quit so communicate returns
  proc.stdin.write("quit\n")
  proc.stdin.flush()


  # Send the input string and get the output
  #stdout, stderr = proc.communicate(input_str)
  stdout, stderr = proc.communicate()
  #print(stdout)

  # Extract the last 2 lines, then the second-to-last line (tail -2 | head -1)
  lines = stdout.strip().splitlines()
  if len(lines) >= 2:
      line = lines[-2]
  else:
      line = lines[-1] if lines else ""

  # Use regex to extract the score cp value
  match = re.search(r"score cp (-?\d{1,3})", line)
  if match:
      score = int(match.group(1))
      #print(score)
  else:
      score = -100
  return score

def get_valid_move_reward(prefix, last_move):
  board = chess.Board()
  for move in prefix.split(' '):
    move = move.strip()
    if move:
      board.push_uci(move)
  #print(board)
  #print(board.legal_moves, last_move)
  try:
    last_move = chess.Move.from_uci(last_move)
  except Exception:
    return -20
  return 1 if last_move in board.legal_moves else -10

def get_reward(response: str, game_prefix) -> float:
    try:
        #print(response)
        if random.random() < 0.01:
          print(game_prefix)
          print(response)
          print()
        given_answer = extract_boxed(response)
        stockfish_query = strfile.format(game_prefix.strip(), given_answer)
        #print(stockfish_query)
        valid_reward = get_valid_move_reward(game_prefix, given_answer)
        #print('v', valid_reward)
        if valid_reward < 0:
          return valid_reward
        reward = call_stockfish(stockfish_query)
        #print(reward)
        #print('s', reward)
        return reward/100
        #ground_truth = extract_gsm8k_final_answer(answer)
        #return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
    except ValueError as e:
        return -30

#def get_reward(response: str, answer: str) -> float:
#    try:
#        given_answer = extract_boxed(response)
#        ground_truth = extract_gsm8k_final_answer(answer)
#        return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
#    except ValueError:
#        return 0.0


def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load GSM8K dataset
    logger.info("Loading dataset...")
    #dataset = datasets.load_dataset("openai/gsm8k", "main")
    dataset = datasets.load_from_disk("/home/johnhew/code/chess-uci/lichess_move_prefixes")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]

    question_suffix = "Provide an answer in uci notation, like e2e4, written inside \\boxed{}."

    chess_game_ex = """
    d2d4 g8f6 c2c4 e7e6 g1f3 d7d5 b1c3 f8e7 g2g3 e8g8 f1g2 d5c4 f3e5 c7c5 d4c5 d8d1 c3d1 e7c5 e1g1 b8c6 g2c6 b7c6 c1e3 c5b6

r . b . . r k .
p . . . . p p p
. b p . p n . .
. . . . N . . .
. . p . . . . .
. . . . B . P .
P P . . P P . P
R . . N . R K .
"""

    chess_discussion_ex = r"""
    White is down a pawn but has active minors: a knight on e5 pesters f7/c6 and the bishop on e3 eyes the b6 bishop. Black’s extra pawns (c6, c4, e6) are a bit loose; the c4 pawn is the most vulnerable. The cleanest way to equalize material and simplify Black’s queenside pressure is to pick off that pawn immediately.

If 13 e5c4, Black’s most principled try is 13…b6e3 14 d1e3, when material is level and White keeps two knights versus Black’s bishop+knight; the c6 pawn can then be targeted with a2a4–a4a5 or f1c1/d1c3. If Black prefers 13…c8a6 to harass the c4 knight, White can shore up with b2b3 and follow with a1c1 or d1c3. An alternative is 13 e3b6 a7b6 14 e5c4, but capturing on c4 first avoids giving Black the useful a-pawn recapture and keeps more control.

\boxed{e5c4}
"""

    q_prefix = "Consider the following prefix of a chess game (and corresponding chess board for your convenience.) Give the best next move continuation. "

    convo_prefix = [
        {
            "role": "user",
            "content": q_prefix + chess_game_ex + question_suffix,
        },
        {
            "role": "assistant",
            "content": chess_discussion_ex,
        },
    ]

    n_train_batches = len(train_dataset) // config.batch_size
    n_train_batches = 21

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    # Optimizer step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(f"Training for {n_train_batches} batches")

    #  Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        # Setup metrics for logging
        t_start = time.time()
        step = batch_idx
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get training batch and convert to datums online
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)
        # Set up sampling parameters

        training_datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_futures: list[list[Future[types.SampleResponse]]] = []
        batch_prompts: list[list[int]] = []
        for game_prefix, board_state in zip(batch_rows["uci_prefix"], batch_rows['board_ascii']):
            question = q_prefix 
            question = question + '\n' + game_prefix + '\n' + board_state
            #print(f"\n  {i+1}. UCI: {prefix_dataset[i]['uci_prefix']}")
            #print(f"     Board:")
            #print(prefix_dataset[i]['board_ascii'])
            convo = [
                *convo_prefix,
                {"role": "user", "content": question + question_suffix},
            ]
            model_input = renderer.build_generation_prompt(convo)
            prompt_tokens = model_input.to_ints()

            # Generate response
            sample_futures: list[Future[types.SampleResponse]] = []
            for _ in range(config.group_size):
                sample_futures.append(
                    sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                )

            batch_futures.append(sample_futures)
            batch_prompts.append(prompt_tokens)

        #for sample_futures, prompt_tokens, answer in zip(
        #    batch_futures, batch_prompts, batch_rows["answer"]
        #):
        for sample_futures, prompt_tokens, game_prefix in zip(
            batch_futures, batch_prompts, batch_rows['uci_prefix']
        ):
            group_rewards: list[float] = []
            group_tokens: list[list[int]] = []
            group_logprobs: list[list[float]] = []
            group_ob_lens: list[int] = []
            for future in sample_futures:
                sample_result = future.result()
                sampled_tokens = sample_result.sequences[0].tokens
                sampled_logprobs = sample_result.sequences[0].logprobs
                assert sampled_logprobs is not None

                all_tokens = prompt_tokens + sampled_tokens
                group_tokens.append(all_tokens)
                group_ob_lens.append(len(prompt_tokens) - 1)
                group_logprobs.append(sampled_logprobs)

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                reward = get_reward(parsed_message["content"], game_prefix)
                group_rewards.append(reward)

            advantages = [
                reward - (sum(group_rewards) / len(group_rewards)) for reward in group_rewards
            ]
            print(group_rewards)
            batch_rewards.append(sum(group_rewards) / len(group_rewards))

            # check if all advantages are zero
            if all(advantage == 0.0 for advantage in advantages):
                # Skip question because all advantages are the same
                continue

            for tokens, logprob, advantage, ob_len in zip(
                group_tokens, group_logprobs, advantages, group_ob_lens
            ):
                input_tokens = tokens[:-1]
                input_tokens = [int(token) for token in input_tokens]
                target_tokens = tokens[1:]
                all_logprobs = [0.0] * ob_len + logprob
                all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
                assert (
                    len(input_tokens)
                    == len(target_tokens)
                    == len(all_logprobs)
                    == len(all_advantages)
                ), (
                    f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, len(all_logprobs): {len(all_logprobs)}, len(all_advantages): {len(all_advantages)}"
                )
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)

        # Training step
        fwd_bwd_future = training_client.forward_backward(
            training_datums, loss_fn="importance_sampling"
        )
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Log metrics[]
        metrics["time/total"] = time.time() - t_start
        metrics["reward/mean"] = sum(batch_rewards) / len(batch_rewards)
        ml_logger.log_metrics(metrics, step=batch_idx)

        # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
