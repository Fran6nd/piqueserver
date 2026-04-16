"""
Example bot script using the piqueserver Bot API.

Spawns one guard bot per team.  Each guard:

* Walks toward the nearest enemy using physics-driven movement
* Stops and aims when it has line-of-sight
* Shoots when in range and line-of-sight is clear
* Throws a grenade when the enemy is very close
* Recovers from being stuck: tries to jump first, then crouches if the jump
  failed or was blocked
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
from abc import ABC, abstractmethod

from piqueserver.bot import Bot, BotManagerMixin
from piqueserver.commands import command
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
# State machine
# ---------------------------------------------------------------------------

class GuardState(ABC):
    """Abstract base for GuardBot FSM states."""

    def enter(self) -> None:
        """Called once when this state becomes active."""

    @abstractmethod
    def think(self, bot: 'GuardBot', dt: float) -> 'GuardState':
        """
        Called every tick.  Return ``self`` to stay in this state or a new
        ``GuardState`` instance to transition.
        """


class RoamState(GuardState):
    """No visible enemies — sprint toward the enemy tent."""

    def think(self, bot: 'GuardBot', dt: float) -> GuardState:
        visible = [e for e in bot.get_enemies() if bot.can_see(e)]
        if visible:
            target = bot.closest(visible)
            dist = bot.distance_to(target)
            return EngageState() if dist <= bot.shoot_range else PursueState()

        conn = bot.connection
        enemy_base = getattr(getattr(conn.team, 'other', None), 'base', None)
        if enemy_base is None or bot.distance_to((enemy_base.x, enemy_base.y, enemy_base.z)) < 3.0:
            bot.set_walk()
            return self

        dest = (enemy_base.x, enemy_base.y, enemy_base.z)
        bot.look_horizontal_toward(dest)
        jump, crouch = bot._try_unstick(dt)
        bot.set_walk(up=True, sprint=True, jump=jump, crouch=crouch)
        return self


class PursueState(GuardState):
    """Visible enemy out of shoot range — close the distance."""

    def think(self, bot: 'GuardBot', dt: float) -> GuardState:
        visible = [e for e in bot.get_enemies() if bot.can_see(e)]
        if not visible:
            return RoamState()

        target = bot.closest(visible)
        if bot.distance_to(target) <= bot.shoot_range:
            return EngageState()

        bot.look_horizontal_toward(target)
        jump, crouch = bot._try_unstick(dt)
        bot.set_walk(up=True, sprint=True, jump=jump, crouch=crouch)
        return self


class EngageState(GuardState):
    """Visible enemy in shoot range — stand, aim, and fire."""

    def think(self, bot: 'GuardBot', _dt: float) -> GuardState:
        visible = [e for e in bot.get_enemies() if bot.can_see(e)]
        if not visible:
            return RoamState()

        target = bot.closest(visible)
        dist = bot.distance_to(target)
        if dist > bot.shoot_range:
            return PursueState()

        bot.look_toward(target)
        bot.set_walk()

        if dist <= bot.grenade_range and bot._grenade_cooldown <= 0:
            vel = bot._grenade_velocity_toward(target)
            if bot.throw_grenade(fuse=2.5, velocity=vel) is not None:
                bot._grenade_cooldown = 5.0

        if bot._shoot_cooldown <= 0:
            bot.shoot_at(target)
            bot._shoot_cooldown = bot.connection.weapon_object.delay

        return self


# ---------------------------------------------------------------------------
# GuardBot
# ---------------------------------------------------------------------------

class GuardBot(Bot):
    """
    A guard bot that engages the nearest visible enemy.

    States
    ------
    RoamState   — no visible enemies, walk toward the enemy tent
    PursueState — visible enemy out of shoot range, sprint toward it
    EngageState — visible enemy in shoot range, stand still and fire

    Obstacle recovery
    -----------------
    While moving, ``_try_unstick`` checks every ``_STUCK_CHECK_INTERVAL``
    seconds whether the bot has made horizontal progress.  If it has not,
    it attempts a two-step recovery:

    1. **Jump** — if the block directly above is clear, issue a one-shot jump.
       This clears small ledges without physics hacks.
    2. **Crouch** — if the bot is still stuck on the next check (jump didn't
       help) *or* if jumping is blocked (solid block above), hold crouch for
       ``_UNSTICK_CROUCH_DURATION`` seconds.  The reduced hitbox lets the bot
       slide under low ceilings or through narrow gaps.

    ``_try_unstick`` returns ``(jump, crouch)`` booleans that the calling
    state merges into its ``set_walk`` call so only one broadcast happens per
    tick.
    """

    # Stuck-detection tunables
    _STUCK_CHECK_INTERVAL: float = 0.5   # seconds between position checks
    _STUCK_THRESHOLD: float = 0.8        # min blocks moved to be un-stuck
    _UNSTICK_JUMP_GRACE: float = 1.0     # seconds to wait after a jump before escalating to crouch
    _UNSTICK_CROUCH_DURATION: float = 1.5  # seconds to hold crouch after failed jump

    # How often to force-resend InputData so clients never miss the walk state
    _INPUT_REFRESH_INTERVAL: float = 1.0

    shoot_range: float
    grenade_range: float
    _shoot_cooldown: float
    _grenade_cooldown: float
    _low_hp_warned: bool
    _state: GuardState
    _stuck_timer: float
    _pos_at_check: tuple
    _input_refresh_timer: float
    _unstick_jump_tried: bool
    _unstick_jump_grace: float
    _unstick_crouch_remaining: float

    def __init_bot__(self) -> None:
        self.shoot_range = float(_shoot_range_opt.get())
        self.grenade_range = float(_grenade_range_opt.get())
        self._shoot_cooldown = 0.0
        self._grenade_cooldown = 0.0
        self._low_hp_warned = False
        self._state = RoamState()
        self._stuck_timer = 0.0
        self._pos_at_check = (0.0, 0.0, 0.0)
        self._input_refresh_timer = 0.0
        self._unstick_jump_tried = False
        self._unstick_jump_grace = 0.0
        self._unstick_crouch_remaining = 0.0

    # ------------------------------------------------------------------
    # AI tick
    # ------------------------------------------------------------------

    def think(self, dt: float) -> None:
        conn = self.connection
        if not conn.hp or conn.world_object is None:
            self._low_hp_warned = False
            return

        self._shoot_cooldown = max(0.0, self._shoot_cooldown - dt)
        self._grenade_cooldown = max(0.0, self._grenade_cooldown - dt)

        # Periodically invalidate the cached walk state so the next set_walk()
        # call always re-broadcasts InputData.  This keeps clients in sync even
        # when the bot stays in the same movement state for a long time.
        self._input_refresh_timer += dt
        if self._input_refresh_timer >= self._INPUT_REFRESH_INTERVAL:
            self._input_refresh_timer = 0.0
            self._walk_state = None

        if conn.hp < 30 and not self._low_hp_warned:
            self._low_hp_warned = True
            self.chat(f'{conn.name} is taking heavy fire!', global_message=False)

        next_state = self._state.think(self, dt)
        if next_state is not self._state:
            self._state = next_state
            self._state.enter()
            # Reset stuck tracker on every state transition so the new state
            # gets a clean baseline position for its first interval check.
            self._stuck_timer = 0.0
            pos = self.position
            if pos is not None:
                self._pos_at_check = pos

    # ------------------------------------------------------------------
    # Obstacle recovery
    # ------------------------------------------------------------------

    def _try_unstick(self, dt: float) -> tuple:
        """
        Detect when stuck and return ``(jump, crouch)`` flags for the caller's
        ``set_walk`` call.

        Called every tick from movement states.  Accumulates ``dt`` and checks
        every ``_STUCK_CHECK_INTERVAL`` seconds.  If the bot moved less than
        ``_STUCK_THRESHOLD`` blocks horizontally it is considered stuck.

        Recovery is two-stage:

        1. **Jump** — if the block directly above (z − 1 in AoS z-down coords)
           is clear, return ``jump=True`` for one tick.  This uses the physics
           engine to hop over small ledges.
        2. **Crouch** — if the jump attempt did not produce progress on the
           next check, *or* if the block above is solid (jump is impossible),
           set ``_unstick_crouch_remaining`` and return ``crouch=True`` for
           ``_UNSTICK_CROUCH_DURATION`` seconds.  The reduced hitbox lets the
           bot slide under low ceilings or through tight gaps.

        Between interval checks the method returns whichever flags are
        currently active so the movement state always gets a consistent value.
        """
        # Tick down active timers every frame.
        if self._unstick_jump_grace > 0.0:
            self._unstick_jump_grace = max(0.0, self._unstick_jump_grace - dt)
        if self._unstick_crouch_remaining > 0.0:
            self._unstick_crouch_remaining = max(
                0.0, self._unstick_crouch_remaining - dt
            )

        pos = self.position
        if pos is None:
            return (False, self._unstick_crouch_remaining > 0.0)

        self._stuck_timer += dt
        if self._stuck_timer < self._STUCK_CHECK_INTERVAL:
            return (False, self._unstick_crouch_remaining > 0.0)
        self._stuck_timer = 0.0

        bx, by, bz = pos
        ox, oy, _ = self._pos_at_check
        moved = math.sqrt((bx - ox) ** 2 + (by - oy) ** 2)
        self._pos_at_check = pos

        if moved >= self._STUCK_THRESHOLD:
            # Making progress — clear jump state, let crouch countdown finish.
            self._unstick_jump_tried = False
            self._unstick_jump_grace = 0.0
            return (False, self._unstick_crouch_remaining > 0.0)

        # Stuck.  Decide which recovery action to take.
        if self.connection.world_object is None:
            return (False, self._unstick_crouch_remaining > 0.0)

        # If a jump was just issued, give it time to produce horizontal movement
        # before deciding it failed.  The grace window covers the arc of the jump
        # so that a successful hop over a ledge is not misread as "still stuck".
        if self._unstick_jump_grace > 0.0:
            return (False, self._unstick_crouch_remaining > 0.0)

        ix, iy, iz = int(bx), int(by), int(bz)
        map_ = self.protocol.map

        # Walking auto-steps 1-block walls, so reaching the stuck threshold
        # always means a 2-block obstacle.  We need 2 clear blocks above to
        # hop over it: iz-1 (just above head) and iz-2 (the landing space).
        above1_clear = not map_.get_solid(ix, iy, iz - 1)
        above2_clear = above1_clear and not map_.get_solid(ix, iy, iz - 2)

        # "On ground" means there is solid terrain within 2-3 blocks below.
        # iz+1 sits inside the player body in AoS z-down coords; iz+2/iz+3
        # reliably catches the floor block regardless of exact foot position.
        on_ground = map_.get_solid(ix, iy, iz + 2) or map_.get_solid(ix, iy, iz + 3)

        wo = self.connection.world_object
        if above1_clear and not self._unstick_jump_tried:
            self._unstick_jump_tried = True
            self._unstick_jump_grace = self._UNSTICK_JUMP_GRACE
            if wo is not None:
                if above2_clear:
                    # 2-block wall: jump + 2-block nudge (AoS jump alone only
                    # reaches ~1.35 blocks, not enough to clear).
                    wo.set_position(bx, by, bz - 2.0)
                else:
                    # 1-block wall: pure physics jump, no nudge needed.
                    # AoS jump height is enough to clear a single block.
                    pass
            return (True, False)

        if not on_ground:
            # Bot is in water or airborne — crouching would make it sink/fall.
            # Reset so a jump is retried on the next check interval.
            self._unstick_jump_tried = False
            return (False, False)

        # Stage 2: grace expired, on solid ground, jump didn't help — crouch
        # to reduce hitbox and slide under the obstacle.
        self._unstick_jump_tried = False
        self._unstick_crouch_remaining = self._UNSTICK_CROUCH_DURATION
        return (False, True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_enemies(self) -> list:
        """Return all living enemy players, including enemy bots."""
        conn = self.connection
        if conn.team is None or conn.team.spectator:
            return []
        return [
            p for p in self.protocol.players.values()
            if p.team is not conn.team
            and not p.team.spectator
            and p.hp
            and p.world_object is not None
        ]

    def _grenade_velocity_toward(self, target) -> tuple:
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

        def on_player_connect(self, player):
            # A new client just finished connecting.  Invalidate every bot's
            # cached walk state so the next think() tick re-sends InputData
            # and the new player sees correct walk animations immediately.
            for bot in self.bots:
                bot._walk_state = None
            return super().on_player_connect(player)

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
