"""
Example bot script using the piqueserver Bot API.

Spawns one guard bot per team.  Each guard:

* Walks toward the nearest enemy using physics-driven movement
* Stops and aims when it has line-of-sight
* Shoots when in range and line-of-sight is clear
* Throws a grenade when the enemy is very close
* Builds a quick cover wall in front of itself when taking fire
* Calls for help in chat when its HP drops below 30

This script is intended as a reference for the Bot API.  Remove or adapt
any behaviour you do not need.

Commands
^^^^^^^^

* ``/guards`` — show the current HP of both guard bots (admin only)

Options
^^^^^^^

.. code-block:: toml

   [guardbot]
   # Weapon for both guards: 0 = rifle, 1 = smg, 2 = shotgun
   weapon = 0
   # Distance (blocks) at which the bot starts shooting
   shoot_range = 24
   # Distance (blocks) at which the bot throws a grenade
   grenade_range = 12

.. codeauthor:: piqueserver contributors
"""

import math

from piqueserver.bot import Bot, BotManagerMixin
from piqueserver.commands import command, admin
from piqueserver.config import config

from pyspades.constants import RIFLE_WEAPON, SMG_WEAPON, SHOTGUN_WEAPON

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_section = config.section('guardbot')
_weapon_opt = _section.option('weapon', default=0)
_shoot_range_opt = _section.option('shoot_range', default=24)
_grenade_range_opt = _section.option('grenade_range', default=12)

_WEAPONS = {0: RIFLE_WEAPON, 1: SMG_WEAPON, 2: SHOTGUN_WEAPON}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@command('guards', admin_only=True)
def cmd_guards(connection):
    """Show HP of all guard bots."""
    protocol = connection.protocol
    guards = [b for b in getattr(protocol, 'bots', []) if isinstance(b, GuardBot)]
    if not guards:
        return 'No guard bots active.'
    parts = []
    for g in guards:
        hp = g.connection.hp or 0
        name = g.connection.name or '?'
        parts.append(f'{name}: {hp} HP')
    return ', '.join(parts)


# ---------------------------------------------------------------------------
# GuardBot
# ---------------------------------------------------------------------------

class GuardBot(Bot):
    """
    A simple guard bot that engages the nearest enemy.

    State machine
    -------------
    * No enemies in range → hold position
    * Enemy visible       → stop, aim, shoot (grenade if very close)
    * Enemy not visible   → walk toward enemy
    * Stuck while walking → cycle: jump (0.4 s) → crouch (0.4 s) → repeat
    * HP < 30             → broadcast a help message (once per life)
    """

    _shoot_range: float
    _grenade_range: float
    _shoot_cooldown: float
    _grenade_cooldown: float
    _low_hp_warned: bool

    # Stuck-detection tunables
    _STUCK_CHECK_INTERVAL: float = 0.5   # seconds between position checks
    _STUCK_THRESHOLD: float = 0.8        # min blocks moved to be considered un-stuck
    _UNSTICK_JUMP_INTERVAL: float = 0.5  # min seconds between jump retries when stuck

    def __init_bot__(self) -> None:
        self._shoot_range = float(_shoot_range_opt.get())
        self._grenade_range = float(_grenade_range_opt.get())
        self._shoot_cooldown = 0.0
        self._grenade_cooldown = 0.0
        self._low_hp_warned = False
        self._reset_stuck()

    # ------------------------------------------------------------------
    # AI tick
    # ------------------------------------------------------------------

    def think(self, dt: float) -> None:
        conn = self.connection
        if not conn.hp or conn.world_object is None:
            # Dead — reset per-life state when the bot respawns next tick
            self._low_hp_warned = False
            self._reset_stuck()
            return

        self._shoot_cooldown = max(0.0, self._shoot_cooldown - dt)
        self._grenade_cooldown = max(0.0, self._grenade_cooldown - dt)

        # Low-HP warning (once per life)
        if conn.hp < 30 and not self._low_hp_warned:
            self._low_hp_warned = True
            self.chat(f'{conn.name} is taking heavy fire!', global_message=False)

        # Only consider enemies we can actually see
        visible = [e for e in self.get_enemies() if self.can_see(e)]
        if not visible:
            self._roam(dt)
            return

        target = self.closest(visible)
        dist = self.distance_to(target)

        self.look_toward(target)

        if dist <= self._shoot_range:
            # In range — stand still and engage
            self.set_walk()
            self._reset_stuck()

            # Grenade when enemy is very close
            if dist <= self._grenade_range and self._grenade_cooldown <= 0:
                vel = self._grenade_velocity_toward(target)
                if self.throw_grenade(fuse=2.5, velocity=vel) is not None:
                    self._grenade_cooldown = 5.0

            # Shoot
            if self._shoot_cooldown <= 0:
                self.shoot_at(target)
                self._shoot_cooldown = conn.weapon_object.delay

        else:
            # Visible but out of shoot range — close the distance
            self._walk_toward(target, dt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _walk_toward(self, target, dt: float) -> None:
        """
        Orient toward target and walk forward, with automatic stuck recovery.

        Every ``_STUCK_CHECK_INTERVAL`` seconds the bot measures how far it
        has moved horizontally.  If it moved less than ``_STUCK_THRESHOLD``
        blocks it is considered stuck and enters a recovery loop:

        When stuck, every ``_UNSTICK_JUMP_INTERVAL`` seconds the bot checks
        whether there is a solid block directly above it:

        * No block above → jump (clear a 2-block ledge)
        * Block above    → crouch (squeeze through a low gap; jumping would
          just hit the ceiling)

        The check repeats each interval until the position check shows progress.
        """
        # Use horizontal orientation so p->f.x/p->f.y are at full magnitude.
        # A pitched view (enemy above/below) reduces those components, making
        # mf=True produce little horizontal thrust and jump appear in-place.
        self.look_horizontal_toward(target)

        # --- Position check ---
        pos = self.position
        if pos is not None:
            self._pos_check_timer += dt
            if self._pos_at_check is None:
                self._pos_at_check = pos

            if self._pos_check_timer >= self._STUCK_CHECK_INTERVAL:
                ox, oy, _ = self._pos_at_check
                nx, ny, _ = pos
                moved = math.sqrt((nx - ox) ** 2 + (ny - oy) ** 2)
                if moved < self._STUCK_THRESHOLD:
                    # Not enough progress — start unsticking if not already
                    if self._unstick_phase < 0:
                        self._unstick_phase = 0
                        self._unstick_timer = 0.0
                        self._fire_jump = True  # one-shot jump on entry
                else:
                    # Moving freely — clear unstick state
                    self._unstick_phase = -1
                    self._unstick_timer = 0.0
                self._pos_check_timer = 0.0
                self._pos_at_check = pos

        # --- Unstick ---
        # _unstick_phase == 0: jumping  (no block above)
        # _unstick_phase == 1: crouching (block above — jump would hit ceiling)
        jump = False
        crouch = False
        if self._unstick_phase >= 0:
            self._unstick_timer += dt
            wo = self.connection.world_object
            grounded = wo is not None and not wo.airborne
            if grounded and self._unstick_timer >= self._UNSTICK_JUMP_INTERVAL:
                self._unstick_timer = 0.0
                if self._has_block_above():
                    self._unstick_phase = 1  # blocked above — crouch
                else:
                    self._unstick_phase = 0  # clear above — jump
                    self._fire_jump = True

            # Hold crouch continuously while in crouch mode
            if self._unstick_phase == 1:
                crouch = True

            jump = self._fire_jump
            self._fire_jump = False

        self.set_walk(up=True, sprint=True, jump=jump, crouch=crouch)

    def _has_block_above(self) -> bool:
        """
        True if there is a solid block close above the bot.

        In AoS, Z increases downward (z=0 is the sky).  The player reference
        point p->p.z sits roughly at waist height; the head reaches ~z-1.35.
        Checking z-1 and z-2 (one and two blocks above the reference point)
        reliably detects a ceiling that would prevent an effective jump.
        """
        pos = self.position
        if pos is None:
            return False
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        map_ = self.protocol.map
        return map_.get_solid(x, y, z - 1) or map_.get_solid(x, y, z - 2)

    def _roam(self, dt: float) -> None:
        """Walk toward the enemy tent when there is nothing to shoot."""
        conn = self.connection
        enemy_base = getattr(getattr(conn.team, 'other', None), 'base', None)
        if enemy_base is None:
            self.set_walk()
            self._reset_stuck()
            return
        dest = (enemy_base.x, enemy_base.y, enemy_base.z)
        if self.distance_to(dest) < 3.0:
            # Arrived — hold position at the tent
            self.set_walk()
            self._reset_stuck()
            return
        self._walk_toward(dest, dt)

    def _reset_stuck(self) -> None:
        """Clear all stuck-detection state (call when stopping or on respawn)."""
        self._pos_check_timer: float = 0.0
        self._pos_at_check = None
        self._unstick_phase: int = -1   # -1 = not stuck; 0 = jump; 1 = crouch
        self._unstick_timer: float = 0.0
        self._fire_jump: bool = False   # True means one jump will fire next tick

    def _grenade_velocity_toward(
        self, target
    ) -> tuple:
        """Compute a modest throw velocity aimed at the target."""
        p1 = self.position
        if p1 is None or target.world_object is None:
            return (0.0, 0.0, -0.3)
        tx, ty, tz = target.world_object.position.get()
        dx, dy, dz = tx - p1[0], ty - p1[1], tz - p1[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 0.001:
            return (0.0, 0.0, -0.3)
        speed = 0.6
        return (dx / length * speed, dy / length * speed, dz / length * speed)


# ---------------------------------------------------------------------------
# apply_script
# ---------------------------------------------------------------------------

def apply_script(protocol, connection, config):
    class GuardBotProtocol(BotManagerMixin, protocol):
        def on_map_change(self, map_):
            # BotManagerMixin.on_map_change removes old bots before calling super.
            # super() triggers set_map() which clears self.players and resets all
            # connections after on_map_change returns, so bots cannot be created
            # inline here — they would be wiped immediately.
            # reactor.callLater(0, ...) defers creation to the next event loop
            # iteration, after set_map() has fully completed.
            super().on_map_change(map_)
            from twisted.internet import reactor
            reactor.callLater(0, self._spawn_guards)

        def _spawn_guards(self):
            weapon = _WEAPONS.get(_weapon_opt.get(), RIFLE_WEAPON)
            for i in range(5):
                self.add_bot(
                    GuardBot.create(self, f'Guard-Blue-{i}', self.team_1, weapon)
                )
                self.add_bot(
                    GuardBot.create(self, f'Guard-Green-{i}', self.team_2, weapon)
                )

    return GuardBotProtocol, connection
