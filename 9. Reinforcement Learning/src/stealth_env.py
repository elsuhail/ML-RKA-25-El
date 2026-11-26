"""Custom Gymnasium environment for Stealth Heist 2D."""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# Action constants
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4

ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
    ACTION_STAY: (0, 0),
}

ACTION_NAMES: Dict[int, str] = {
    ACTION_UP: "Up",
    ACTION_DOWN: "Down",
    ACTION_LEFT: "Left",
    ACTION_RIGHT: "Right",
    ACTION_STAY: "Stay (Hide)",
}

# 11x11 Stealth Heist arena:
# '#' = wall/pillar (blocks movement and flashlight LOS)
# 'S' = agent start (9, 1)
# 'L' = loot (2, 5)
# 'E' = exit (9, 9)
DEFAULT_GRID_LAYOUT = [
    "###########",
    "#.........#",
    "#..#.L.#..#",
    "#.........#",
    "#..#...#..#",
    "#.........#",
    "#..#...#..#",
    "#.........#",
    "#..#...#..#",
    "#S.......E#",
    "###########",
]

# Guard 1 cyclic patrol loop around the upper-center loot zone (16 steps)
GUARD_1_PATH: List[Tuple[int, int]] = [
    (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (2, 8), (1, 8), (1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (2, 2),
]

# Guard 2 cyclic patrol loop across the middle-lower corridors (16 steps)
GUARD_2_PATH: List[Tuple[int, int]] = [
    (7, 8), (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (7, 2),
    (6, 2), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 8),
]


def _bresenham_line(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    """Return grid cells along the line from (r0, c0) to (r1, c1)."""
    cells: List[Tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


class StealthHeistEnv(gym.Env):
    """2D Stealth Heist environment following the Gymnasium API.

    The agent must navigate an 11x11 guarded room, collect the Loot ('L'),
    and escape through the Exit ('E') without colliding with guards or
    entering their occluded 4-tile flashlight cones.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 6}

    def __init__(
        self,
        reward_mode: str = "shaped",
        max_steps: int = 120,
        gamma: float = 0.98,
        shaping_scale: float = 1.5,
        randomize_guard_phase: bool = True,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if reward_mode not in ("shaped", "sparse"):
            raise ValueError("reward_mode must be 'shaped' or 'sparse'")

        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.gamma = gamma
        self.shaping_scale = shaping_scale
        self.randomize_guard_phase = randomize_guard_phase
        self.render_mode = render_mode

        # Parse grid layout
        self.grid_rows = len(DEFAULT_GRID_LAYOUT)
        self.grid_cols = len(DEFAULT_GRID_LAYOUT[0])
        self.walls: Set[Tuple[int, int]] = set()
        self.start_pos: Tuple[int, int] = (9, 1)
        self.loot_pos: Tuple[int, int] = (2, 5)
        self.exit_pos: Tuple[int, int] = (9, 9)

        for r, row_str in enumerate(DEFAULT_GRID_LAYOUT):
            for c, ch in enumerate(row_str):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == "S":
                    self.start_pos = (r, c)
                elif ch == "L":
                    self.loot_pos = (r, c)
                elif ch == "E":
                    self.exit_pos = (r, c)

        self.guard_paths: List[List[Tuple[int, int]]] = [GUARD_1_PATH, GUARD_2_PATH]

        # Precompute BFS distance maps to Loot and Exit
        self.dist_to_loot = self._compute_bfs_distances(self.loot_pos)
        self.dist_to_exit = self._compute_bfs_distances(self.exit_pos)
        self.loot_to_exit_dist = float(self.dist_to_exit[self.loot_pos])

        # Precompute flashlight LOS lookup table: (r, c, dr, dc) -> frozenset of lit cells
        self._los_cache: Dict[Tuple[int, int, int, int], frozenset] = {}
        self._precompute_los_table()

        # Precompute safe initial guard phases so start_pos is never lit at step 0
        self._safe_phase_pairs: List[Tuple[int, int]] = self._compute_safe_phase_pairs()

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(22,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # Episode state variables
        self.agent_pos: Tuple[int, int] = self.start_pos
        self.has_loot: bool = False
        self.guard_indices: List[int] = [0, 0]
        self.steps: int = 0
        self.last_action: int = ACTION_STAY
        self.last_event: str = "START"

    def _compute_bfs_distances(self, target: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        """Compute shortest walkable path distances from all free cells to target."""
        dists: Dict[Tuple[int, int], int] = {target: 0}
        queue = deque([target])
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.grid_rows
                    and 0 <= nc < self.grid_cols
                    and (nr, nc) not in self.walls
                    and (nr, nc) not in dists
                ):
                    dists[(nr, nc)] = dists[(r, c)] + 1
                    queue.append((nr, nc))
        return dists

    def _compute_flashlight_cells(
        self, pos: Tuple[int, int], facing: Tuple[int, int], fov_range: int = 3
    ) -> Set[Tuple[int, int]]:
        """Compute occluded flashlight cone cells from a guard's position and direction."""
        gr, gc = pos
        dr, dc = facing
        perp_r, perp_c = -dc, dr
        lit: Set[Tuple[int, int]] = set()

        for dist in range(1, fov_range + 1):
            lateral_offsets = [0] if dist == 1 else [-1, 0, 1]
            for lat in lateral_offsets:
                tr = gr + dist * dr + lat * perp_r
                tc = gc + dist * dc + lat * perp_c
                if not (0 <= tr < self.grid_rows and 0 <= tc < self.grid_cols):
                    continue
                if (tr, tc) in self.walls:
                    continue
                # Raycast occlusion check: intermediate wall blocks LOS
                line = _bresenham_line(gr, gc, tr, tc)
                blocked = any(cell in self.walls for cell in line[1:-1])
                if not blocked:
                    lit.add((tr, tc))
        return lit

    def _precompute_los_table(self) -> None:
        """Precompute flashlight cones for all guard patrol positions and directions."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for path in self.guard_paths:
            for pos in path:
                for d in directions:
                    key = (pos[0], pos[1], d[0], d[1])
                    if key not in self._los_cache:
                        self._los_cache[key] = frozenset(self._compute_flashlight_cells(pos, d))

    def _get_guard_state(self, guard_idx: int, step_offset: int = 0) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return (position, facing_direction) for a guard at current index + offset."""
        path = self.guard_paths[guard_idx]
        n = len(path)
        idx = (self.guard_indices[guard_idx] + step_offset) % n
        next_idx = (idx + 1) % n
        pos = path[idx]
        nxt = path[next_idx]
        facing = (nxt[0] - pos[0], nxt[1] - pos[1])
        return pos, facing

    def _get_danger_cells(self, step_offset: int = 0) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """Return (guard_positions, lit_flashlight_cells) for a given step offset."""
        guard_cells: Set[Tuple[int, int]] = set()
        lit_cells: Set[Tuple[int, int]] = set()
        for g in range(len(self.guard_paths)):
            pos, facing = self._get_guard_state(g, step_offset=step_offset)
            guard_cells.add(pos)
            key = (pos[0], pos[1], facing[0], facing[1])
            if key in self._los_cache:
                lit_cells.update(self._los_cache[key])
            else:
                lit_cells.update(self._compute_flashlight_cells(pos, facing))
        return guard_cells, lit_cells

    def _compute_safe_phase_pairs(self) -> List[Tuple[int, int]]:
        """Filter guard initial phase pairs so start_pos is guaranteed safe at step 0."""
        safe_pairs: List[Tuple[int, int]] = []
        n0, n1 = len(self.guard_paths[0]), len(self.guard_paths[1])
        for i0 in range(n0):
            for i1 in range(n1):
                self.guard_indices = [i0, i1]
                g_cells, lit_cells = self._get_danger_cells(step_offset=0)
                if self.start_pos not in g_cells and self.start_pos not in lit_cells:
                    safe_pairs.append((i0, i1))
        self.guard_indices = [0, 0]
        return safe_pairs

    def _potential(self, pos: Tuple[int, int], has_loot: bool) -> float:
        """Continuous BFS potential function Phi(s) across the Loot transition."""
        if not has_loot:
            return -(float(self.dist_to_loot[pos]) + self.loot_to_exit_dist)
        return -float(self.dist_to_exit[pos])

    def _get_obs(self) -> np.ndarray:
        """Build 22-dimensional normalized observation vector in [-1, 1]."""
        ar, ac = self.agent_pos
        max_r = float(self.grid_rows - 1)
        max_c = float(self.grid_cols - 1)

        target = self.exit_pos if self.has_loot else self.loot_pos
        tr, tc = target

        # Local 4-direction navigation/wall sensors: -1.0 = wall, +1.0 = closer to target via BFS, 0.0 = walkable other
        dist_map = self.dist_to_exit if self.has_loot else self.dist_to_loot
        curr_dist = dist_map[(ar, ac)]
        wall_sensors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = ar + dr, ac + dc
            if (nr, nc) in self.walls:
                wall_sensors.append(-1.0)
            elif dist_map[(nr, nc)] < curr_dist:
                wall_sensors.append(1.0)
            else:
                wall_sensors.append(0.0)

        # Guard features: relative position and facing direction
        g1_pos, g1_dir = self._get_guard_state(0, step_offset=0)
        g2_pos, g2_dir = self._get_guard_state(1, step_offset=0)

        # 5 local danger sensors for each candidate action (current OR next guard step)
        curr_g, curr_lit = self._get_danger_cells(step_offset=0)
        next_g, next_lit = self._get_danger_cells(step_offset=1)
        all_threat = curr_g | curr_lit | next_g | next_lit

        danger_sensors: List[float] = []
        for a in range(5):
            dr, dc = ACTION_DELTAS[a]
            nr, nc = ar + dr, ac + dc
            if (nr, nc) in self.walls:
                nr, nc = ar, ac
            # If stepping onto Exit with loot, agent wins before guards move (only curr_threat matters)
            if self.has_loot and (nr, nc) == self.exit_pos:
                is_danger = (nr, nc) in (curr_g | curr_lit)
            else:
                is_danger = (nr, nc) in all_threat
            danger_sensors.append(1.0 if is_danger else 0.0)

        obs = np.array(
            [
                ar / max_r,
                ac / max_c,
                1.0 if self.has_loot else 0.0,
                (tr - ar) / max_r,
                (tc - ac) / max_c,
                *wall_sensors,
                (g1_pos[0] - ar) / max_r,
                (g1_pos[1] - ac) / max_c,
                float(g1_dir[0]),
                float(g1_dir[1]),
                (g2_pos[0] - ar) / max_r,
                (g2_pos[1] - ac) / max_c,
                float(g2_dir[0]),
                float(g2_dir[1]),
                *danger_sensors,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset episode state and optionally randomize initial guard patrol phases."""
        super().reset(seed=seed)

        self.agent_pos = self.start_pos
        self.has_loot = False
        self.steps = 0
        self.last_action = ACTION_STAY
        self.last_event = "START"

        if options and "guard_indices" in options:
            self.guard_indices = list(options["guard_indices"])
        elif self.randomize_guard_phase:
            pair_idx = int(self.np_random.integers(0, len(self._safe_phase_pairs)))
            self.guard_indices = list(self._safe_phase_pairs[pair_idx])
        else:
            self.guard_indices = [0, 0]

        obs = self._get_obs()
        info = self._build_info(raw_reward=0.0, caught=False, success=False)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step using strict 3-phase ordering to prevent swap collisions."""
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        self.last_action = action
        prev_pos = self.agent_pos
        prev_loot = self.has_loot
        phi_prev = self._potential(prev_pos, prev_loot)

        raw_reward = -0.3  # Step penalty
        terminated = False
        truncated = False
        caught = False
        success = False
        event = "MOVE" if action != ACTION_STAY else "HIDE"

        # 1. Agent moves
        dr, dc = ACTION_DELTAS[action]
        nr, nc = prev_pos[0] + dr, prev_pos[1] + dc
        if (nr, nc) in self.walls:
            raw_reward -= 0.5  # Wall bump penalty
            event = "WALL_BUMP"
        else:
            self.agent_pos = (nr, nc)

        # 2. Detection check Phase 1 (against current guards/flashlights -> prevents swap collision)
        curr_g, curr_lit = self._get_danger_cells(step_offset=0)
        if self.agent_pos in curr_g or self.agent_pos in curr_lit:
            caught = True
            terminated = True
            raw_reward += -100.0
            event = "CAUGHT"

        # 3. Objective check (Loot & Exit) if not caught
        if not terminated:
            if not self.has_loot and self.agent_pos == self.loot_pos:
                self.has_loot = True
                raw_reward += 40.0
                event = "LOOT"
            elif self.has_loot and self.agent_pos == self.exit_pos:
                success = True
                terminated = True
                raw_reward += 100.0
                event = "SUCCESS"

        # 4. Guards advance 1 step & Detection check Phase 2 (if episode still active)
        if not terminated:
            for g in range(len(self.guard_paths)):
                self.guard_indices[g] = (self.guard_indices[g] + 1) % len(self.guard_paths[g])
            new_g, new_lit = self._get_danger_cells(step_offset=0)
            if self.agent_pos in new_g or self.agent_pos in new_lit:
                caught = True
                terminated = True
                raw_reward += -100.0
                event = "CAUGHT"

        # 5. Truncation check (max_steps)
        if not terminated and self.steps >= self.max_steps:
            truncated = True
            raw_reward += -10.0
            event = "TIMEOUT"

        # Reward shaping (strictly 0.0 when caught; continuous BFS potential otherwise)
        if self.reward_mode == "shaped" and not caught:
            phi_curr = self._potential(self.agent_pos, self.has_loot)
            shaping = self.shaping_scale * (self.gamma * phi_curr - phi_prev)
        else:
            shaping = 0.0

        total_reward = float(raw_reward + shaping)
        self.last_event = event

        obs = self._get_obs()
        info = self._build_info(raw_reward=float(raw_reward), caught=caught, success=success)
        return obs, total_reward, terminated, truncated, info

    def _build_info(self, raw_reward: float, caught: bool, success: bool) -> dict:
        """Build standardized info dict."""
        return {
            "agent_pos": self.agent_pos,
            "has_loot": self.has_loot,
            "steps": self.steps,
            "raw_reward": raw_reward,
            "caught": caught,
            "success": success,
            "event": self.last_event,
        }

    def get_render_state(self) -> dict:
        """Return snapshot of all entities and lit cells for rendering."""
        guards = [self._get_guard_state(g, step_offset=0) for g in range(len(self.guard_paths))]
        _, lit_cells = self._get_danger_cells(step_offset=0)
        return {
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "walls": self.walls,
            "start_pos": self.start_pos,
            "loot_pos": self.loot_pos,
            "exit_pos": self.exit_pos,
            "agent_pos": self.agent_pos,
            "has_loot": self.has_loot,
            "guards": guards,
            "lit_cells": lit_cells,
            "steps": self.steps,
            "last_action": self.last_action,
            "last_event": self.last_event,
        }

    def render(self) -> Optional[np.ndarray]:
        """Return lightweight RGB array representation when render_mode='rgb_array'."""
        if self.render_mode != "rgb_array":
            return None
        tile = 24
        img = np.full((self.grid_rows * tile, self.grid_cols * tile, 3), (24, 28, 36), dtype=np.uint8)
        state = self.get_render_state()

        def fill_cell(pos: Tuple[int, int], color: Tuple[int, int, int], pad: int = 1) -> None:
            r, c = pos
            r0, r1 = r * tile + pad, (r + 1) * tile - pad
            c0, c1 = c * tile + pad, (c + 1) * tile - pad
            img[r0:r1, c0:c1] = color

        for w in state["walls"]:
            fill_cell(w, (70, 78, 92), pad=0)
        for lc in state["lit_cells"]:
            fill_cell(lc, (180, 155, 50), pad=1)
        if not state["has_loot"]:
            fill_cell(state["loot_pos"], (255, 200, 30), pad=4)
        exit_color = (50, 210, 100) if state["has_loot"] else (90, 120, 100)
        fill_cell(state["exit_pos"], exit_color, pad=2)
        for g_pos, _ in state["guards"]:
            fill_cell(g_pos, (220, 60, 60), pad=3)
        agent_color = (80, 220, 255) if not state["has_loot"] else (120, 255, 160)
        fill_cell(state["agent_pos"], agent_color, pad=4)
        return img
