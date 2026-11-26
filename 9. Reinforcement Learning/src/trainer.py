"""Training and evaluation pipeline for Stealth Heist RL."""

import json
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .dqn_agent import BFSHeuristicAgent, DoubleDQNAgent, RandomAgent
from .stealth_env import ACTION_STAY, StealthHeistEnv
from .visualizer import render_episode_gif


def evaluate_agent(
    env: StealthHeistEnv,
    agent,
    seeds: Iterable[int],
    deterministic: bool = True,
) -> Dict[str, Optional[float]]:
    """Evaluate any agent across a fixed sequence of seeds using fair objective metrics."""
    seed_list = list(seeds)
    n_eps = len(seed_list)
    successes = 0
    loots = 0
    caughts = 0
    timeouts = 0
    raw_rewards: List[float] = []
    success_steps: List[int] = []
    total_steps = 0
    hide_steps = 0

    for s in seed_list:
        obs, _ = env.reset(seed=int(s))
        ep_raw = 0.0
        picked_loot = False
        for _ in range(env.max_steps):
            action = agent.select_action(obs, deterministic=deterministic)
            if action == ACTION_STAY:
                hide_steps += 1
            total_steps += 1

            obs, _, term, trunc, info = env.step(action)
            ep_raw += info["raw_reward"]
            if info["has_loot"]:
                picked_loot = True

            if term or trunc:
                if info["success"]:
                    successes += 1
                    success_steps.append(info["steps"])
                elif info["caught"]:
                    caughts += 1
                else:
                    timeouts += 1
                break

        if picked_loot:
            loots += 1
        raw_rewards.append(ep_raw)

    return {
        "episodes": n_eps,
        "success_rate": round(100.0 * successes / n_eps, 2),
        "loot_pickup_rate": round(100.0 * loots / n_eps, 2),
        "caught_rate": round(100.0 * caughts / n_eps, 2),
        "timeout_rate": round(100.0 * timeouts / n_eps, 2),
        "avg_raw_reward": round(float(np.mean(raw_rewards)), 2),
        "avg_steps_success": round(float(np.mean(success_steps)), 2) if success_steps else None,
        "hide_usage_pct": round(100.0 * hide_steps / max(1, total_steps), 2),
    }


def train_double_dqn(
    reward_mode: str = "shaped",
    total_episodes: int = 3000,
    seed: int = 42,
    checkpoint_path: Optional[str] = None,
    milestone_dir: Optional[str] = None,
    milestone_seed: int = 2000,
    verbose: bool = True,
) -> Tuple[DoubleDQNAgent, dict]:
    """Train DoubleDQNAgent with periodic best-checkpoint validation on seeds 1000..1029."""
    env = StealthHeistEnv(reward_mode=reward_mode, randomize_guard_phase=True)
    val_env = StealthHeistEnv(reward_mode="shaped", randomize_guard_phase=True)
    agent = DoubleDQNAgent(seed=seed)

    val_seeds = range(1000, 1030)
    best_score = (-1.0, -1e9)  # (val_success_rate, val_avg_raw_reward)

    history: dict = {
        "raw_rewards": [],
        "total_rewards": [],
        "successes": [],
        "caughts": [],
        "losses": [],
        "val_episodes": [],
        "val_success_rates": [],
        "val_raw_rewards": [],
    }

    t0 = time.time()
    for ep in range(1, total_episodes + 1):
        agent.update_epsilon(ep - 1, total_episodes)
        obs, _ = env.reset(seed=seed + ep)

        ep_raw = 0.0
        ep_total = 0.0
        ep_losses: List[float] = []

        for _ in range(env.max_steps):
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, term, trunc, info = env.step(action)
            loss = agent.step_update(obs, action, reward, next_obs, term)
            if loss is not None:
                ep_losses.append(loss)

            ep_raw += info["raw_reward"]
            ep_total += reward
            obs = next_obs

            if term or trunc:
                history["successes"].append(1 if info["success"] else 0)
                history["caughts"].append(1 if info["caught"] else 0)
                break

        history["raw_rewards"].append(ep_raw)
        history["total_rewards"].append(ep_total)
        history["losses"].append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        # Record milestone GIFs without slowing down training
        if milestone_dir and ep == 1:
            render_episode_gif(
                val_env, agent, os.path.join(milestone_dir, "episode_early.gif"),
                seed=milestone_seed, deterministic=True, title="Fase Awal (Episode 1)"
            )
        elif milestone_dir and ep == min(500, total_episodes // 3):
            render_episode_gif(
                val_env, agent, os.path.join(milestone_dir, "episode_mid.gif"),
                seed=milestone_seed, deterministic=True, title=f"Fase Tengah (Episode {ep})"
            )

        # Periodic validation & Best Checkpoint saving (every 50 episodes)
        if ep % 50 == 0 or ep == total_episodes:
            val_metrics = evaluate_agent(val_env, agent, val_seeds, deterministic=True)
            history["val_episodes"].append(ep)
            history["val_success_rates"].append(val_metrics["success_rate"])
            history["val_raw_rewards"].append(val_metrics["avg_raw_reward"])

            score = (val_metrics["success_rate"], val_metrics["avg_raw_reward"])
            if score > best_score:
                best_score = score
                if checkpoint_path:
                    agent.save(checkpoint_path)

            if verbose and (ep % 500 == 0 or ep == total_episodes):
                print(
                    f"[{reward_mode.upper():6s}] Ep {ep:4d}/{total_episodes} | "
                    f"eps={agent.epsilon:.2f} | Val Success={val_metrics['success_rate']:5.1f}% | "
                    f"Val RawRew={val_metrics['avg_raw_reward']:+6.1f} | "
                    f"Elapsed={time.time()-t0:.1f}s"
                )

    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)

    if milestone_dir:
        render_episode_gif(
            val_env, agent, os.path.join(milestone_dir, "episode_trained.gif"),
            seed=milestone_seed, deterministic=True, title="Agen Terlatih (Best Checkpoint)"
        )

    return agent, history


def run_full_experiment(output_dir: str = "outputs", total_episodes: int = 3000) -> dict:
    """Run full Phase 5 experiment: Shaped vs Sparse Double DQN + Random & BFS baselines."""
    os.makedirs(output_dir, exist_ok=True)
    shaped_ckpt = os.path.join(output_dir, "dqn_shaped_best.pth")
    sparse_ckpt = os.path.join(output_dir, "dqn_sparse_best.pth")

    print("=== 1. Training Double DQN (Shaped Reward) ===")
    agent_shaped, hist_shaped = train_double_dqn(
        reward_mode="shaped",
        total_episodes=total_episodes,
        seed=42,
        checkpoint_path=shaped_ckpt,
        milestone_dir=output_dir,
        milestone_seed=2000,
    )

    print("\n=== 2. Training Double DQN (Sparse Reward - Ablation) ===")
    agent_sparse, hist_sparse = train_double_dqn(
        reward_mode="sparse",
        total_episodes=total_episodes,
        seed=42,
        checkpoint_path=sparse_ckpt,
        milestone_dir=None,
    )

    # Evaluate all 4 agents fairly on unseen test seeds 2000..2099
    eval_env = StealthHeistEnv(reward_mode="shaped", randomize_guard_phase=True)
    test_seeds = range(2000, 2100)

    rand_agent = RandomAgent(seed=42)
    bfs_agent = BFSHeuristicAgent(eval_env)

    print("\n=== 3. Evaluating All 4 Agents on 100 Unseen Guard Phases (seeds 2000..2099) ===")
    summary = {
        "Random Agent": evaluate_agent(eval_env, rand_agent, test_seeds, deterministic=True),
        "BFS Heuristic": evaluate_agent(eval_env, bfs_agent, test_seeds, deterministic=True),
        "Double DQN (Sparse)": evaluate_agent(eval_env, agent_sparse, test_seeds, deterministic=True),
        "Double DQN (Shaped)": evaluate_agent(eval_env, agent_shaped, test_seeds, deterministic=True),
    }

    with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "training_histories.json"), "w", encoding="utf-8") as f:
        json.dump({"shaped": hist_shaped, "sparse": hist_sparse}, f)

    for name, m in summary.items():
        print(
            f"{name:22s} | Success={m['success_rate']:5.1f}% | Loot={m['loot_pickup_rate']:5.1f}% | "
            f"Caught={m['caught_rate']:5.1f}% | RawRew={m['avg_raw_reward']:+6.1f} | "
            f"Hide={m['hide_usage_pct']:4.1f}% | AvgSteps={m['avg_steps_success']}"
        )
    return summary
