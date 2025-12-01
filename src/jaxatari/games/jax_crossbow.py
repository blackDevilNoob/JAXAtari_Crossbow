from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'cursor', 'type': 'single', 'file': 'cursor.npy'},
        {'name': 'friend', 'type': 'single', 'file': 'friend.npy'},
        {'name': 'enemy', 'type': 'single', 'file': 'enemy.npy'},
        {'name': 'cactus', 'type': 'single', 'file': 'cactus.npy'},
        {'name': 'mountain', 'type': 'single', 'file': 'mountain.npy'},
        {'name': 'shot', 'type': 'single', 'file': 'shot.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'digit_{}.npy'},
    )

class CrossbowConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210

    # Gameplay
    CURSOR_SPEED: int = 3
    CURSOR_SIZE: Tuple[int, int] = (4, 4)
    FRIEND_SPEED: float = 0.5
    ENEMY_SPEED: float = 0.5
    MAX_ENEMIES: int = 10
    MAX_LIVES: int = 3

    # Dimensions
    FRIEND_SIZE: Tuple[int, int] = (6, 12)
    ENEMY_SIZE: Tuple[int, int] = (8, 8)
    SHOT_SIZE: Tuple[int, int] = (6, 6)
    PLAY_AREA_HEIGHT: int = 190

    ASSET_CONFIG: tuple = _get_default_asset_config()


class CrossbowState(NamedTuple):
    cursor_x: chex.Array
    cursor_y: chex.Array
    is_firing: chex.Array
    friend_x: chex.Array
    friend_y: chex.Array
    friend_active: chex.Array
    enemies_x: chex.Array
    enemies_y: chex.Array
    enemies_active: chex.Array
    score: chex.Array
    lives: chex.Array
    enemy_score: chex.Array
    step_counter: chex.Array
    key: chex.PRNGKey


class CrossbowObservation(NamedTuple):
    cursor_x: jnp.ndarray
    cursor_y: jnp.ndarray
    friend_x: jnp.ndarray
    lives: jnp.ndarray
    score: jnp.ndarray


class CrossbowInfo(NamedTuple):
    time: jnp.ndarray


class JaxCrossbow(JaxEnvironment[CrossbowState, CrossbowObservation, CrossbowInfo, CrossbowConstants]):
    def __init__(self, consts: CrossbowConstants = None):
        consts = consts or CrossbowConstants()
        super().__init__(consts)
        self.renderer = CrossbowRenderer(self.consts)
        self.action_set = Action.get_all_values()

    def _cursor_step(self, state: CrossbowState, action: chex.Array) -> CrossbowState:
        is_up = jnp.isin(action, jnp.array([Action.UP, Action.UPRIGHT, Action.UPLEFT, Action.UPFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE]))
        is_down = jnp.isin(action, jnp.array([Action.DOWN, Action.DOWNRIGHT, Action.DOWNLEFT, Action.DOWNFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE]))
        is_left = jnp.isin(action, jnp.array([Action.LEFT, Action.UPLEFT, Action.DOWNLEFT, Action.LEFTFIRE, Action.UPLEFTFIRE, Action.DOWNLEFTFIRE]))
        is_right = jnp.isin(action, jnp.array([Action.RIGHT, Action.UPRIGHT, Action.DOWNRIGHT, Action.RIGHTFIRE, Action.UPRIGHTFIRE, Action.DOWNRIGHTFIRE]))

        dx = jnp.where(is_left, -self.consts.CURSOR_SPEED, 0)
        dx = jnp.where(is_right, self.consts.CURSOR_SPEED, dx)
        dy = jnp.where(is_up, -self.consts.CURSOR_SPEED, 0)
        dy = jnp.where(is_down, self.consts.CURSOR_SPEED, dy)

        new_x = jnp.clip(state.cursor_x + dx, 0, self.consts.WIDTH - self.consts.CURSOR_SIZE[0])
        new_y = jnp.clip(state.cursor_y + dy, 0, self.consts.PLAY_AREA_HEIGHT - self.consts.CURSOR_SIZE[1])

        return state._replace(cursor_x=new_x.astype(jnp.int32), cursor_y=new_y.astype(jnp.int32))

    def _friend_step(self, state: CrossbowState) -> CrossbowState:
        should_move = state.step_counter % 2 == 0
        new_x = state.friend_x + jnp.where(should_move & state.friend_active, 1, 0)
        reached_goal = new_x > self.consts.WIDTH
        final_x = jnp.where(reached_goal, 0, new_x)
        return state._replace(friend_x=final_x.astype(jnp.int32))

    def _enemy_and_combat_step(self, state: CrossbowState, action: chex.Array) -> Tuple[CrossbowState, bool]:
        rng, spawn_key = jax.random.split(state.key)

        # Combat
        is_fire = jnp.isin(action, jnp.array([
            Action.FIRE, Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE,
            Action.DOWNFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
        ]))

        HIT_TOLERANCE = 8
        cx, cy = state.cursor_x, state.cursor_y
        ex, ey = state.enemies_x, state.enemies_y

        hit_x = jnp.logical_and(cx < ex + self.consts.ENEMY_SIZE[0] + HIT_TOLERANCE, cx + self.consts.CURSOR_SIZE[0] > ex - HIT_TOLERANCE)
        hit_y = jnp.logical_and(cy < ey + self.consts.ENEMY_SIZE[1] + HIT_TOLERANCE, cy + self.consts.CURSOR_SIZE[1] > ey - HIT_TOLERANCE)
        is_hit = jnp.logical_and(hit_x, hit_y)

        valid_kill = jnp.logical_and(is_fire, jnp.logical_and(state.enemies_active, is_hit))
        surviving_enemies = jnp.logical_and(state.enemies_active, jnp.logical_not(valid_kill))

        # Collision
        fx, fy = state.friend_x, state.friend_y
        danger_x = jnp.logical_and(ex < fx + self.consts.FRIEND_SIZE[0], ex + self.consts.ENEMY_SIZE[0] > fx)
        danger_y = jnp.logical_and(ey < fy + self.consts.FRIEND_SIZE[1], ey + self.consts.ENEMY_SIZE[1] > fy)
        is_danger = jnp.logical_and(danger_x, danger_y)
        friend_hit = jnp.logical_and(is_danger, jnp.logical_and(surviving_enemies, state.friend_active))
        any_friend_hit = jnp.any(friend_hit)

        # Updates
        enemies_active_next = surviving_enemies
        friend_x_next = jnp.where(any_friend_hit, 0, state.friend_x).astype(jnp.int32)
        friend_active_next = jnp.array(True)

        # Scoring (1 point per kill)
        score_gain = jnp.sum(valid_kill) * 1
        new_score = state.score + score_gain
        new_enemy_score = state.enemy_score + jnp.where(any_friend_hit, 1, 0)

        new_lives = state.lives - jnp.where(any_friend_hit, 1, 0)
        new_lives = jnp.maximum(0, new_lives)
        is_game_over = jnp.logical_and(any_friend_hit, state.lives == 0)

        # Spawning
        spawn_chance = jax.random.uniform(spawn_key, shape=(self.consts.MAX_ENEMIES,)) < 0.05
        should_spawn = jnp.logical_and(jnp.logical_not(enemies_active_next), spawn_chance)
        spawn_x = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 20, self.consts.WIDTH)
        spawn_y = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 20, 100)
        should_move_enemy = state.step_counter % 3 == 0
        new_enemy_y = state.enemies_y + jnp.where(should_move_enemy, 1, 0)
        final_active = jnp.logical_or(enemies_active_next, should_spawn)
        final_x = jnp.where(should_spawn, spawn_x, state.enemies_x)
        final_y = jnp.where(should_spawn, spawn_y, new_enemy_y)
        final_active = jnp.logical_and(final_active, final_y < self.consts.PLAY_AREA_HEIGHT)

        new_state = state._replace(
            score=new_score.astype(jnp.int32),
            lives=new_lives.astype(jnp.int32),
            enemy_score=new_enemy_score.astype(jnp.int32),
            friend_x=friend_x_next,
            enemies_active=final_active,
            enemies_x=final_x.astype(jnp.int32),
            enemies_y=final_y.astype(jnp.int32),
            is_firing=is_fire,
            key=rng
        )
        return new_state, is_game_over

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[CrossbowObservation, CrossbowState]:
        state_key, _ = jax.random.split(key)
        state = CrossbowState(
            cursor_x=jnp.array(self.consts.WIDTH // 2, dtype=jnp.int32),
            cursor_y=jnp.array(self.consts.HEIGHT // 2, dtype=jnp.int32),
            is_firing=jnp.array(False),
            friend_x=jnp.array(0, dtype=jnp.int32),
            friend_y=jnp.array(150, dtype=jnp.int32),
            friend_active=jnp.array(True),
            enemies_x=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.int32),
            enemies_y=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.int32),
            enemies_active=jnp.zeros(self.consts.MAX_ENEMIES, dtype=bool),
            score=jnp.array(0, dtype=jnp.int32),
            enemy_score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.MAX_LIVES, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            key=state_key
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CrossbowState, action: chex.Array):
        prev_score = state.score

        new_key, step_key = jax.random.split(state.key)
        state = state._replace(key=step_key)

        state = self._cursor_step(state, action)
        state = self._friend_step(state)
        state, game_over = self._enemy_and_combat_step(state, action)
        state = state._replace(step_counter=state.step_counter + 1, key=new_key)

        reward = (state.score - prev_score).astype(float)

        done = jnp.logical_or(game_over, state.step_counter > 4000)

        return self._get_observation(state), state, reward, done, self._get_info(state)

    def _get_reward(self, previous_state: CrossbowState, state: CrossbowState):
        return (state.score - previous_state.score).astype(float)

    def _get_observation(self, state):
        return CrossbowObservation(state.cursor_x, state.cursor_y, state.friend_x, state.lives, state.score)

    def _get_info(self, state):
        return CrossbowInfo(time=state.step_counter)

    def obs_to_flat_array(self, obs):
        return jnp.array([0])

    def action_space(self): return spaces.Discrete(18)
    def observation_space(self): return spaces.Dict({})
    def image_space(self): return spaces.Box(0, 255, (210, 160, 3), jnp.uint8)

    def render(self, state: CrossbowState) -> jnp.ndarray:
        return self.renderer.render(state)


class CrossbowRenderer(JAXGameRenderer):
    def __init__(self, consts: CrossbowConstants = None):
        super().__init__(consts)
        self.consts = consts or CrossbowConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        sprite_path = os.path.join(base_dir, "sprites", "crossbow")

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(self.consts.ASSET_CONFIG, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CrossbowState):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Scenery
        raster = self.jr.render_at(raster, 40, 40, self.SHAPE_MASKS["cactus"])
        raster = self.jr.render_at(raster, 120, 100, self.SHAPE_MASKS["cactus"])
        raster = self.jr.render_at(raster, 100, 30, self.SHAPE_MASKS["mountain"])

        # Friend
        friend_mask = self.SHAPE_MASKS["friend"]
        raster = jax.lax.cond(
            state.friend_active,
            lambda r: self.jr.render_at(r, state.friend_x, state.friend_y, friend_mask),
            lambda r: r,
            raster
        )

        # Enemies
        enemy_mask = self.SHAPE_MASKS["enemy"]
        def _draw_enemy(i, r):
            return jax.lax.cond(
                state.enemies_active[i],
                lambda _r: self.jr.render_at(_r, state.enemies_x[i], state.enemies_y[i], enemy_mask),
                lambda _r: _r,
                r
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_ENEMIES, _draw_enemy, raster)

        raster = self.jr.render_at(raster, state.cursor_x, state.cursor_y, self.SHAPE_MASKS["cursor"])

        raster = jax.lax.cond(
            state.is_firing,
            lambda r: self.jr.render_at(r, state.cursor_x, state.cursor_y, self.SHAPE_MASKS["shot"]),
            lambda r: r,
            raster
        )

        digit_masks = self.SHAPE_MASKS["digits"]

        enemy_score_digits = self.jr.int_to_digits(state.enemy_score, max_digits=2)
        raster = self.jr.render_label(raster, 10, 195, enemy_score_digits, digit_masks, spacing=5, max_digits=2)

        player_score_digits = self.jr.int_to_digits(state.score, max_digits=4)
        raster = self.jr.render_label(raster, 90, 195, player_score_digits, digit_masks, spacing=5, max_digits=4)

        lives_digits = self.jr.int_to_digits(state.lives, max_digits=2)
        raster = self.jr.render_label(raster, 140, 195, lives_digits, digit_masks, spacing=6, max_digits=2)

        return self.jr.render_from_palette(raster, self.PALETTE)