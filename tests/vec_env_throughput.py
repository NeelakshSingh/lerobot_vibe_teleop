import argparse
import json
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pufferlib
import pufferlib.vector as pv
from pufferlib.emulation import GymnasiumPufferEnv

# Import package to ensure "LeRobot-v0" gets registered via __init__.py
import lerobothackathonenv  # noqa: F401


def _run_benchmark(vec_env, num_envs: int, num_steps: int) -> tuple[int, float, float, float]:
    obs, info = vec_env.reset()

    total_steps = 0
    start_time = time.time()

    while total_steps < num_steps:
        actions = vec_env.action_space.sample()
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        batch_size = len(obs)
        total_steps += batch_size

    elapsed = time.time() - start_time
    steps_per_second = total_steps / elapsed if elapsed > 0 else float("inf")
    per_env_steps_per_second = steps_per_second / num_envs
    return total_steps, elapsed, steps_per_second, per_env_steps_per_second


def measure_throughput(
    env_id: str = "LeRobot-v0",
    num_envs: int = 4,
    num_steps: int = 1_000_000,
    backend: str = "gym",
    num_workers_pl: int=4,
    verbose: bool = True,
) -> tuple[int, float, float, float]:
    """
    Run a simple rollout with random actions in a vectorized env backend and
    report throughput.
    """
    if backend == "gym":
        env_fns = [lambda _=None: gym.make(env_id) for _ in range(num_envs)]
        vec_env = gym.vector.AsyncVectorEnv(env_fns)
    elif backend == "puffer":
        workers = min(num_envs, num_workers_pl)

        vec_env = pv.make(
            GymnasiumPufferEnv,
            env_args=None,
            env_kwargs={"env_creator": gym.make, "env_args": [env_id]},
            backend=pv.Multiprocessing,
            num_envs=num_envs,
            num_workers=workers,
            batch_size=num_envs,
        )
    else:
        raise ValueError(f"Unknown backend '{backend}', use 'gym' or 'puffer'.")

    try:
        total_steps, elapsed, steps_per_second, per_env_steps_per_second = _run_benchmark(
            vec_env, num_envs, num_steps
        )
    finally:
        vec_env.close()

    if verbose:
        print(f"Vector backend    : {backend}")
        print(f"Env id            : {env_id}")
        print(f"Num envs          : {num_envs}")
        print(f"Total env steps   : {total_steps}")
        print(f"Elapsed time [s]  : {elapsed:.6f}")
        print(f"Throughput [steps/s]       : {steps_per_second:.2f}")
        print(
            "Per-env throughput [steps/s/env] : "
            f"{per_env_steps_per_second:.2f}"
        )

    return total_steps, elapsed, steps_per_second, per_env_steps_per_second


def sweep_and_plot(
    env_id: str,
    num_envs_list: list[int],
    num_steps: int,
    backend: str,
    num_workers_pl: int,
    plot_path: str,
) -> None:
    throughputs = []
    jsonl_path = plot_path.rsplit(".", 1)[0] + ".jsonl"

    with open(jsonl_path, "w") as jf:
        for n in num_envs_list:
            print(f"Running benchmark for num_envs={n}...")
            _, _, steps_per_second, _ = measure_throughput(
                env_id=env_id,
                num_envs=n,
                num_steps=num_steps,
                backend=backend,
                num_workers_pl=min(n, num_workers_pl),
                verbose=False,
            )
            throughputs.append(steps_per_second)
            jf.write(
                json.dumps(
                    {
                        "env_id": env_id,
                        "backend": backend,
                        "num_envs": n,
                        "num_workers": min(n, num_workers_pl),
                        "steps_per_second": steps_per_second,
                    }
                )
                + "\n"
            )
            print(f"  -> {steps_per_second:.2f} steps/s")

    plt.figure()
    plt.plot(num_envs_list, throughputs, marker="o")
    plt.xlabel("num_envs")
    plt.ylabel("env steps per second (total)")
    plt.title(f"Throughput vs num_envs ({backend}, {env_id})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved throughput plot to '{plot_path}'.")
    print(f"Saved JSONL log to '{jsonl_path}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep num_envs and plot throughput of LeRobot-v0 "
            "with a Gymnasium vector env."
        ),
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="LeRobot-v0",
        help="Gymnasium environment id to benchmark.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="(Unused) Kept for backward compatibility.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1_000_000,
        help="Total number of environment steps across all envs.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["gym", "puffer"],
        default="gym",
        help="Vector backend: gym AsyncVectorEnv or pufferlib Multiprocessing.",
    )
    parser.add_argument(
        "--puffer-num-workers",
        type=str,
        default="4",
        help=(
            "PufferLib worker settings. Can be a single int "
            "(e.g. '4') or a comma-separated list (e.g. '2,4,8'). "
            "Effective workers per run are min(num_envs, workers). "
            "Ignored when backend != 'puffer'."
        ),
    )
    parser.add_argument(
        "--sweep-num-envs",
        type=str,
        default="",
        help=(
            "Comma-separated list of num_envs values to sweep and plot "
            "(e.g. '4,8,16,24'). If empty, defaults to '4,8,16,24'."
        ),
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="images",
        help="Output directory for throughput-vs-num_envs plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sweep_num_envs:
        num_envs_list = [
            int(x) for x in args.sweep_num_envs.split(",") if x.strip()
        ]
    else:
        num_envs_list = [4, 8, 16, 24]

    # Treat plot-path as an output directory
    output_dir = args.plot_path
    os.makedirs(output_dir, exist_ok=True)

    # Parse puffer worker settings into a list of ints
    try:
        worker_values = [
            int(x) for x in args.puffer_num_workers.split(",") if x.strip()
        ]
    except ValueError:
        raise ValueError(
            f"Invalid --puffer-num-workers value: {args.puffer_num_workers!r}"
        )

    if args.backend == "puffer" and len(worker_values) > 1:
        base_name = "throughput_puffer"
        for w in worker_values:
            filename = f"{base_name}_num_w_{w}.png"
            plot_path = os.path.join(output_dir, filename)
            sweep_and_plot(
                env_id=args.env_id,
                num_envs_list=num_envs_list,
                num_steps=args.num_steps,
                backend=args.backend,
                num_workers_pl=w,
                plot_path=plot_path,
            )
    else:
        # Single setting (gym or puffer)
        workers = worker_values[0]
        base_name = f"throughput_{args.backend}"
        filename = f"{base_name}.png"
        plot_path = os.path.join(output_dir, filename)
        sweep_and_plot(
            env_id=args.env_id,
            num_envs_list=num_envs_list,
            num_steps=args.num_steps,
            backend=args.backend,
            num_workers_pl=workers,
            plot_path=plot_path,
        )


if __name__ == "__main__":
    main()
