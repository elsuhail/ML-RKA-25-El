"""Visualization utilities for Stealth Heist (Matplotlib 3.11+ & Pillow)."""

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .stealth_env import ACTION_NAMES, StealthHeistEnv


def _fig_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    """Convert Matplotlib figure to RGB uint8 array safely (Matplotlib 3.11+)."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def draw_arena_frame(ax: plt.Axes, state: dict, title: str = "") -> None:
    """Draw a single 2D Stealth Heist arena frame onto the given Matplotlib Axes."""
    rows, cols = state["grid_rows"], state["grid_cols"]
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#151922")
    ax.set_xticks([])
    ax.set_yticks([])

    # Subtle grid lines
    for x in np.arange(-0.5, cols, 1.0):
        ax.axvline(x, color="#232938", linewidth=0.6, zorder=1)
    for y in np.arange(-0.5, rows, 1.0):
        ax.axhline(y, color="#232938", linewidth=0.6, zorder=1)

    # Walls and cover pillars
    for r, c in state["walls"]:
        rect = patches.Rectangle(
            (c - 0.5, r - 0.5), 1.0, 1.0,
            facecolor="#384152", edgecolor="#4B5563", linewidth=0.8, zorder=2
        )
        ax.add_patch(rect)

    # Flashlight cones (semi-transparent warm yellow)
    for r, c in state["lit_cells"]:
        rect = patches.Rectangle(
            (c - 0.48, r - 0.48), 0.96, 0.96,
            facecolor="#FACC15", alpha=0.38, edgecolor="#EAB308", linewidth=0.6, zorder=3
        )
        ax.add_patch(rect)

    # Exit tile
    er, ec = state["exit_pos"]
    exit_fc = "#22C55E" if state["has_loot"] else "#374151"
    exit_ec = "#86EFAC" if state["has_loot"] else "#6B7280"
    ax.add_patch(
        patches.Rectangle(
            (ec - 0.42, er - 0.42), 0.84, 0.84,
            facecolor=exit_fc, edgecolor=exit_ec, linewidth=1.8, zorder=4
        )
    )
    ax.text(ec, er, "EXIT", color="white", ha="center", va="center", fontsize=7, fontweight="bold", zorder=5)

    # Loot tile (shown until picked up)
    if not state["has_loot"]:
        lr, lc = state["loot_pos"]
        ax.scatter([lc], [lr], marker="D", s=160, color="#F59E0B", edgecolors="#FEF3C7", linewidths=1.5, zorder=5)
        ax.text(lc, lr + 0.02, "L", color="#111827", ha="center", va="center", fontsize=7, fontweight="bold", zorder=6)

    # Guards with facing arrows
    for idx, (g_pos, g_dir) in enumerate(state["guards"]):
        gr, gc = g_pos
        dr, dc = g_dir
        ax.scatter([gc], [gr], marker="s", s=180, color="#EF4444", edgecolors="#FCA5A5", linewidths=1.5, zorder=6)
        ax.arrow(
            gc, gr, dc * 0.35, dr * 0.35,
            head_width=0.22, head_length=0.18, fc="#FEF08A", ec="#FEF08A", zorder=7
        )

    # Agent
    ar, ac = state["agent_pos"]
    agent_color = "#34D399" if state["has_loot"] else "#38BDF8"
    agent_edge = "#FBBF24" if state["has_loot"] else "#E0F2FE"
    ax.scatter([ac], [ar], marker="o", s=200, color=agent_color, edgecolors=agent_edge, linewidths=2.0, zorder=8)
    ax.text(ac, ar, "A", color="#0F172A", ha="center", va="center", fontsize=8, fontweight="bold", zorder=9)

    # Status title
    act_name = ACTION_NAMES.get(state["last_action"], "-")
    loot_str = "Loot: YES" if state["has_loot"] else "Loot: NO"
    header = f"{title}\n" if title else ""
    ax.set_title(
        f"{header}Step {state['steps']} | {act_name} | {loot_str} | {state['last_event']}",
        fontsize=9.5, color="#111827", pad=6
    )


def render_episode_gif(
    env: StealthHeistEnv,
    agent,
    gif_path: str,
    seed: int = 42,
    deterministic: bool = True,
    fps: int = 4,
    title: str = "",
) -> Tuple[List[np.ndarray], List[dict], dict]:
    """Run one episode, render frames, export GIF via Pillow, and return trajectory."""
    obs, info = env.reset(seed=seed)
    states_history = [env.get_render_state()]
    frames: List[np.ndarray] = []

    def capture(st: dict) -> np.ndarray:
        fig, ax = plt.subplots(figsize=(5.0, 5.2), dpi=110)
        draw_arena_frame(ax, st, title=title)
        fig.tight_layout()
        arr = _fig_to_rgb_array(fig)
        plt.close(fig)
        return arr

    frames.append(capture(states_history[-1]))

    for _ in range(env.max_steps):
        action = agent.select_action(obs, deterministic=deterministic)
        obs, _, term, trunc, info = env.step(action)
        st = env.get_render_state()
        states_history.append(st)
        frames.append(capture(st))
        if term or trunc:
            break

    # Hold final frame briefly so outcome is easy to read
    for _ in range(5):
        frames.append(frames[-1])

    pil_frames = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / max(1, fps))
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return frames, states_history, info


def plot_storyboard(
    states_history: List[dict],
    title: str = "Storyboard Perjalanan Agen",
    max_frames: int = 6,
    save_path: Optional[str] = None,
) -> None:
    """Plot evenly spaced key frames from an episode as a static storyboard."""
    n = len(states_history)
    if n <= max_frames:
        indices = list(range(n))
    else:
        # Always include start, first HIDE, LOOT pickup, and final frame if present
        key_indices = {0, n - 1}
        for i, st in enumerate(states_history):
            if st["last_event"] == "LOOT":
                key_indices.add(i)
                break
        for i, st in enumerate(states_history):
            if st["last_event"] == "HIDE" and i > 0:
                key_indices.add(i)
                break
        # Fill remaining slots evenly across the episode
        candidates = np.linspace(0, n - 1, max_frames, dtype=int).tolist()
        for idx in candidates:
            if len(key_indices) >= max_frames:
                break
            key_indices.add(int(idx))
        indices = sorted(key_indices)[:max_frames]

    ncols = len(indices)
    fig, axes = plt.subplots(1, ncols, figsize=(3.3 * ncols, 3.8))
    if ncols == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        draw_arena_frame(ax, states_history[idx])

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.show()
    plt.close(fig)


def plot_learning_curves(
    histories: Dict[str, dict],
    window: int = 50,
    save_path: Optional[str] = None,
) -> None:
    """Plot moving-average Raw Reward, Success Rate, Caught Rate, and TD Loss."""
    def moving_avg(vals: List[float], w: int) -> np.ndarray:
        arr = np.asarray(vals, dtype=np.float32)
        if len(arr) < w:
            return arr
        kernel = np.ones(w, dtype=np.float32) / w
        return np.convolve(arr, kernel, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = ["#2563EB", "#DC2626", "#16A34A", "#9333EA"]

    metrics = [
        ("raw_rewards", "Raw Objective Reward (Moving Avg)", "Reward"),
        ("successes", "Success Rate (Moving Avg %)", "Success Rate (%)"),
        ("caughts", "Caught Rate (Moving Avg %)", "Caught Rate (%)"),
        ("losses", "TD Loss per Episode (Moving Avg)", "Huber Loss"),
    ]

    for ax, (key, m_title, ylabel) in zip(axes.flatten(), metrics):
        for idx, (label, hist) in enumerate(histories.items()):
            vals = hist.get(key, [])
            if not vals:
                continue
            y = moving_avg(vals, window)
            if key in ("successes", "caughts"):
                y = y * 100.0
            x = np.arange(len(y)) + (len(vals) - len(y) + 1)
            ax.plot(x, y, label=label, color=colors[idx % len(colors)], linewidth=2.0)

        ax.set_title(m_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.show()
    plt.close(fig)


def plot_value_and_policy_map(
    agent,
    guard_indices: Tuple[int, int] = (2, 10),
    has_loot: bool = False,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot V(s) = max Q(s, a) heatmap and greedy policy arrows on an isolated env."""
    iso_env = StealthHeistEnv(randomize_guard_phase=False)
    iso_env.reset(options={"guard_indices": list(guard_indices)})
    iso_env.has_loot = has_loot

    rows, cols = iso_env.grid_rows, iso_env.grid_cols
    value_grid = np.full((rows, cols), np.nan, dtype=np.float32)
    policy_grid = np.full((rows, cols), -1, dtype=int)

    for r in range(rows):
        for c in range(cols):
            if (r, c) in iso_env.walls:
                continue
            iso_env.agent_pos = (r, c)
            obs = iso_env._get_obs()
            q_vals = agent.get_q_values(obs)
            value_grid[r, c] = float(np.max(q_vals))
            policy_grid[r, c] = int(np.argmax(q_vals))

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#2D3748")

    im = ax.imshow(value_grid, cmap=cmap, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("State Value V(s) = max Q(s, a)")

    # Overlay policy arrows
    arrow_deltas = {0: (0, -0.28), 1: (0, 0.28), 2: (-0.28, 0), 3: (0.28, 0)}
    for r in range(rows):
        for c in range(cols):
            act = policy_grid[r, c]
            if act == -1:
                continue
            if act == 4:
                ax.scatter([c], [r], marker="o", s=35, color="#1E293B", zorder=4)
            else:
                dx, dy = arrow_deltas[act]
                ax.arrow(c - dx * 0.35, r - dy * 0.35, dx, dy, head_width=0.18, head_length=0.14, fc="#0F172A", ec="#0F172A", zorder=4)

    # Overlay guard snapshot and flashlight cone
    g_cells, lit_cells = iso_env._get_danger_cells(step_offset=0)
    for lr, lc in lit_cells:
        ax.add_patch(
            patches.Rectangle(
                (lc - 0.48, lr - 0.48), 0.96, 0.96,
                facecolor="none", edgecolor="#FACC15", linewidth=2.0, linestyle="--", zorder=5
            )
        )
    for g_idx in range(2):
        (gr, gc), (dr, dc) = iso_env._get_guard_state(g_idx, step_offset=0)
        ax.scatter([gc], [gr], marker="s", s=160, color="#DC2626", edgecolors="white", linewidths=1.5, zorder=6)
        ax.arrow(gc, gr, dc * 0.32, dr * 0.32, head_width=0.2, head_length=0.15, fc="yellow", ec="black", zorder=7)

    # Loot and Exit markers
    if not has_loot:
        ax.scatter([iso_env.loot_pos[1]], [iso_env.loot_pos[0]], marker="D", s=140, color="#F59E0B", edgecolors="black", zorder=6)
    ax.scatter([iso_env.exit_pos[1]], [iso_env.exit_pos[0]], marker="*", s=210, color="#16A34A", edgecolors="white", zorder=6)

    phase_str = "Menuju Harta (has_loot=0)" if not has_loot else "Menuju Exit (has_loot=1)"
    ax.set_title(title or f"Value Heatmap V(s) & Policy Arrows\n{phase_str}", fontsize=11, fontweight="bold")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.show()
    plt.close(fig)
