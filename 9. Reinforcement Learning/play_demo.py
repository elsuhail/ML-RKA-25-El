"""Standalone Pygame Live Demo for Stealth Heist (Hybrid Visualization).

Controls:
  1 : AI Double DQN (trained model)
  2 : Random Agent
  3 : BFS Heuristic Agent (shortest path, ignores guards)
  4 : Manual Mode (WASD / Arrow Keys to move, Space to Stay/Hide)
  R : Reset Episode
  + / - : Speed up / slow down simulation FPS
  Esc : Quit
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pygame

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.dqn_agent import BFSHeuristicAgent, DoubleDQNAgent, RandomAgent
from src.stealth_env import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NAMES,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    StealthHeistEnv,
)

TILE_SIZE = 54
HUD_WIDTH = 340
GRID_SIZE = 11
WINDOW_W = GRID_SIZE * TILE_SIZE + HUD_WIDTH
WINDOW_H = GRID_SIZE * TILE_SIZE


def _load_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Load system font with safe fallback to Pygame default font."""
    try:
        return pygame.font.SysFont("consolas", size, bold=bold)
    except Exception:
        return pygame.font.Font(None, size)


def draw_game(
    screen: pygame.Surface,
    env: StealthHeistEnv,
    mode: str,
    q_values: Optional[np.ndarray],
    chosen_action: int,
    total_raw_reward: float,
    sim_fps: int,
    ai_available: bool,
    notice: str,
    done_state: str,
    fonts: dict,
) -> None:
    """Render the 2D Stealth Heist arena and live Q-Value HUD onto Pygame surface."""
    screen.fill((18, 22, 30))
    state = env.get_render_state()

    # 1. Draw grid cells
    for r in range(state["grid_rows"]):
        for c in range(state["grid_cols"]):
            rect = pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if (r, c) in state["walls"]:
                pygame.draw.rect(screen, (55, 65, 81), rect)
                pygame.draw.rect(screen, (75, 85, 99), rect, 1)
            else:
                pygame.draw.rect(screen, (24, 29, 40), rect)
                pygame.draw.rect(screen, (35, 42, 56), rect, 1)

    # 2. Draw flashlight cone
    cone_surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    cone_surf.fill((250, 204, 21, 105))
    for lr, lc in state["lit_cells"]:
        screen.blit(cone_surf, (lc * TILE_SIZE, lr * TILE_SIZE))

    # 3. Draw Exit
    er, ec = state["exit_pos"]
    exit_rect = pygame.Rect(ec * TILE_SIZE + 5, er * TILE_SIZE + 5, TILE_SIZE - 10, TILE_SIZE - 10)
    exit_col = (34, 197, 94) if state["has_loot"] else (65, 85, 75)
    pygame.draw.rect(screen, exit_col, exit_rect, border_radius=6)
    exit_lbl = fonts["small"].render("EXIT", True, (255, 255, 255))
    screen.blit(exit_lbl, exit_lbl.get_rect(center=exit_rect.center))

    # 4. Draw Loot
    if not state["has_loot"]:
        lr, lc = state["loot_pos"]
        cx, cy = lc * TILE_SIZE + TILE_SIZE // 2, lr * TILE_SIZE + TILE_SIZE // 2
        pts = [(cx, cy - 16), (cx + 16, cy), (cx, cy + 16), (cx - 16, cy)]
        pygame.draw.polygon(screen, (245, 158, 11), pts)
        pygame.draw.polygon(screen, (254, 243, 199), pts, 2)

    # 5. Draw Guards & facing triangle
    for (gr, gc), (dr, dc) in state["guards"]:
        g_rect = pygame.Rect(gc * TILE_SIZE + 8, gr * TILE_SIZE + 8, TILE_SIZE - 16, TILE_SIZE - 16)
        pygame.draw.rect(screen, (239, 68, 68), g_rect, border_radius=6)
        cx, cy = gc * TILE_SIZE + TILE_SIZE // 2, gr * TILE_SIZE + TILE_SIZE // 2
        pygame.draw.circle(screen, (254, 240, 138), (cx + dc * 15, cy + dr * 15), 5)

    # 6. Draw Agent
    ar, ac = state["agent_pos"]
    ax, ay = ac * TILE_SIZE + TILE_SIZE // 2, ar * TILE_SIZE + TILE_SIZE // 2
    a_col = (52, 211, 153) if state["has_loot"] else (56, 189, 248)
    pygame.draw.circle(screen, a_col, (ax, ay), 17)
    ring_col = (251, 191, 36) if state["has_loot"] else (224, 242, 254)
    pygame.draw.circle(screen, ring_col, (ax, ay), 17, 3)

    # 7. Draw HUD Panel
    hud_x = GRID_SIZE * TILE_SIZE
    pygame.draw.rect(screen, (15, 18, 25), (hud_x, 0, HUD_WIDTH, WINDOW_H))
    pygame.draw.line(screen, (55, 65, 81), (hud_x, 0), (hud_x, WINDOW_H), 2)

    y = 16
    screen.blit(fonts["title"].render("STEALTH HEIST RL", True, (248, 250, 252)), (hud_x + 16, y))
    y += 32

    mode_colors = {"AI": (56, 189, 248), "RANDOM": (248, 113, 113), "BFS": (251, 191, 36), "MANUAL": (52, 211, 153)}
    screen.blit(fonts["med"].render(f"Mode : {mode}", True, mode_colors.get(mode, (255, 255, 255))), (hud_x + 16, y))
    y += 24
    screen.blit(fonts["small"].render(f"Step : {state['steps']}/{env.max_steps}   Speed: {sim_fps} FPS", True, (203, 213, 225)), (hud_x + 16, y))
    y += 22
    loot_col = (52, 211, 153) if state["has_loot"] else (251, 191, 36)
    screen.blit(fonts["small"].render(f"Loot : {'CAPTURED' if state['has_loot'] else 'NOT YET'}", True, loot_col), (hud_x + 16, y))
    y += 22
    screen.blit(fonts["small"].render(f"Raw Reward: {total_raw_reward:+.1f}", True, (203, 213, 225)), (hud_x + 16, y))
    y += 22
    screen.blit(fonts["small"].render(f"Action: {ACTION_NAMES.get(chosen_action, '-')}", True, (203, 213, 225)), (hud_x + 16, y))
    y += 28

    status_col = (34, 197, 94) if done_state == "SUCCESS!" else ((239, 68, 68) if "CAUGHT" in done_state or "TIME" in done_state else (148, 163, 184))
    screen.blit(fonts["med"].render(f"Status: {done_state}", True, status_col), (hud_x + 16, y))
    y += 32

    # Live Q-Value Bars
    pygame.draw.line(screen, (45, 55, 72), (hud_x + 16, y), (hud_x + HUD_WIDTH - 16, y), 1)
    y += 10
    screen.blit(fonts["med"].render("Live Q-Values (AI Brain):", True, (226, 232, 240)), (hud_x + 16, y))
    y += 26

    if q_values is not None:
        max_abs = max(10.0, float(np.max(np.abs(q_values))))
        best_a = int(np.argmax(q_values))
        bar_center_x = hud_x + 175
        half_w = 75

        for a_idx in range(5):
            q = float(q_values[a_idx])
            is_best = (a_idx == best_a)
            lbl_col = (250, 204, 21) if is_best else (148, 163, 184)
            short_name = ACTION_NAMES[a_idx].split(" ")[0]
            screen.blit(fonts["small"].render(f"{'>' if is_best else ' '} {short_name:5s}", True, lbl_col), (hud_x + 16, y))

            # Track background and center zero-axis
            pygame.draw.rect(screen, (30, 41, 59), (bar_center_x - half_w, y + 3, half_w * 2, 13), border_radius=3)
            pygame.draw.line(screen, (100, 116, 139), (bar_center_x, y + 1), (bar_center_x, y + 18), 1)

            bar_len = int((abs(q) / max_abs) * half_w)
            bar_col = (52, 211, 153) if q >= 0 else (248, 113, 113)
            if q >= 0:
                pygame.draw.rect(screen, bar_col, (bar_center_x, y + 3, bar_len, 13), border_radius=2)
            else:
                pygame.draw.rect(screen, bar_col, (bar_center_x - bar_len, y + 3, bar_len, 13), border_radius=2)

            if is_best:
                pygame.draw.rect(screen, (250, 204, 21), (bar_center_x - half_w, y + 3, half_w * 2, 13), 1, border_radius=3)

            q_txt = fonts["small"].render(f"{q:+.1f}", True, lbl_col)
            screen.blit(q_txt, (hud_x + 258, y))
            y += 26
    else:
        msg = "Load outputs/dqn_shaped_best.pth" if not ai_available else "Q-Values inactive in this mode"
        screen.blit(fonts["small"].render(msg, True, (148, 163, 184)), (hud_x + 16, y))
        y += 130

    if notice:
        y += 8
        screen.blit(fonts["small"].render(notice, True, (251, 191, 36)), (hud_x + 16, y))

    # Controls footer
    ctrl_y = WINDOW_H - 132
    pygame.draw.line(screen, (45, 55, 72), (hud_x + 16, ctrl_y - 8), (hud_x + HUD_WIDTH - 16, ctrl_y - 8), 1)
    controls = [
        "[1] AI Double DQN   [2] Random",
        "[3] BFS Heuristic   [4] Manual",
        "[WASD/Arrows] Move  [Space] Hide",
        "[R] Reset   [+ / -] Speed",
        "[Esc] Quit",
    ]
    for line in controls:
        screen.blit(fonts["small"].render(line, True, (148, 163, 184)), (hud_x + 16, ctrl_y))
        ctrl_y += 22


def run_demo(smoke_test: bool = False) -> None:
    """Launch Pygame interactive viewer (or run headless smoke test)."""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Stealth Heist - Deep RL Live Demo")

    fonts = {
        "title": _load_font(20, bold=True),
        "med": _load_font(15, bold=True),
        "small": _load_font(13, bold=False),
    }

    env = StealthHeistEnv(reward_mode="shaped", randomize_guard_phase=True)
    dqn_agent = DoubleDQNAgent(seed=42)
    rand_agent = RandomAgent(seed=42)
    bfs_agent = BFSHeuristicAgent(env)

    sparse_path = BASE_DIR / "outputs" / "dqn_sparse_best.pth"
    shaped_path = BASE_DIR / "outputs" / "dqn_shaped_best.pth"
    model_path = sparse_path if sparse_path.exists() else shaped_path
    ai_available = model_path.exists()
    notice = ""
    if ai_available:
        dqn_agent.load(str(model_path))
        mode = "AI"
    else:
        mode = "BFS"
        notice = "AI weights not found; fallback to BFS"

    obs, _ = env.reset(seed=42)
    total_raw = 0.0
    chosen_action = ACTION_STAY
    done_state = "RUNNING"
    sim_fps = 5
    clock = pygame.time.Clock()
    step_accum = 0.0

    running = True
    smoke_frames = 0
    while running:
        dt = clock.tick(60) / 1000.0
        manual_action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    if event.key == pygame.K_1:
                        if ai_available:
                            mode = "AI"
                            notice = ""
                        else:
                            notice = "Train notebook first for AI weights!"
                    elif event.key == pygame.K_2:
                        mode = "RANDOM"
                        notice = ""
                    elif event.key == pygame.K_3:
                        mode = "BFS"
                        notice = ""
                    elif event.key == pygame.K_4:
                        mode = "MANUAL"
                        notice = ""
                    # Auto-reset if switching mode after episode already ended
                    if done_state != "RUNNING":
                        obs, _ = env.reset()
                        total_raw = 0.0
                        done_state = "RUNNING"
                        chosen_action = ACTION_STAY
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    total_raw = 0.0
                    done_state = "RUNNING"
                    chosen_action = ACTION_STAY
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    sim_fps = min(20, sim_fps + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    sim_fps = max(1, sim_fps - 1)
                elif mode == "MANUAL" and done_state == "RUNNING":
                    if event.key in (pygame.K_w, pygame.K_UP):
                        manual_action = ACTION_UP
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        manual_action = ACTION_DOWN
                    elif event.key in (pygame.K_a, pygame.K_LEFT):
                        manual_action = ACTION_LEFT
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        manual_action = ACTION_RIGHT
                    elif event.key == pygame.K_SPACE:
                        manual_action = ACTION_STAY

        q_vals = dqn_agent.get_q_values(obs) if ai_available else None

        if done_state == "RUNNING":
            if mode == "MANUAL":
                if manual_action is not None:
                    chosen_action = manual_action
                    obs, _, term, trunc, info = env.step(chosen_action)
                    total_raw += info["raw_reward"]
                    if info["success"]:
                        done_state = "SUCCESS!"
                    elif info["caught"]:
                        done_state = "CAUGHT!"
                    elif trunc:
                        done_state = "TIMEOUT!"
            else:
                step_accum += dt
                if step_accum >= (1.0 / sim_fps) or smoke_test:
                    step_accum = 0.0
                    if mode == "AI" and ai_available:
                        chosen_action = dqn_agent.select_action(obs, deterministic=True)
                    elif mode == "RANDOM":
                        chosen_action = rand_agent.select_action(obs)
                    else:
                        chosen_action = bfs_agent.select_action(obs)

                    obs, _, term, trunc, info = env.step(chosen_action)
                    total_raw += info["raw_reward"]
                    if info["success"]:
                        done_state = "SUCCESS!"
                    elif info["caught"]:
                        done_state = "CAUGHT!"
                    elif trunc:
                        done_state = "TIMEOUT!"

        draw_game(
            screen, env, mode, q_vals, chosen_action,
            total_raw, sim_fps, ai_available, notice, done_state, fonts
        )
        pygame.display.flip()

        if smoke_test:
            smoke_frames += 1
            if smoke_frames >= 5:
                running = False

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stealth Heist Pygame Live Demo")
    parser.add_argument("--smoke-test", action="store_true", help="Run 5 headless frames and exit")
    args = parser.parse_args()
    run_demo(smoke_test=args.smoke_test)
