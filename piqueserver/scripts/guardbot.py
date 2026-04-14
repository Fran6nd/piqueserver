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
    * HP < 30             → broadcast a help message (once per life)
    """

    _shoot_range: float
    _grenade_range: float
    _shoot_cooldown: float
    _grenade_cooldown: float
    _low_hp_warned: bool

    def __init_bot__(self) -> None:
        self._shoot_range = float(_shoot_range_opt.get())
        self._grenade_range = float(_grenade_range_opt.get())
        self._shoot_cooldown = 0.0
        self._grenade_cooldown = 0.0
        self._low_hp_warned = False

    # ------------------------------------------------------------------
    # AI tick
    # ------------------------------------------------------------------

    def think(self, dt: float) -> None:
        conn = self.connection
        if not conn.hp or conn.world_object is None:
            # Dead — reset per-life state when the bot respawns next tick
            self._low_hp_warned = False
            return

        self._shoot_cooldown = max(0.0, self._shoot_cooldown - dt)
        self._grenade_cooldown = max(0.0, self._grenade_cooldown - dt)

        # Low-HP warning (once per life)
        if conn.hp < 30 and not self._low_hp_warned:
            self._low_hp_warned = True
            self.chat(f'{conn.name} is taking heavy fire!', global_message=False)

        enemies = self.get_enemies()
        if not enemies:
            self.set_walk()  # stop
            return

        target = self.closest(enemies)
        dist = self.distance_to(target)

        self.look_toward(target)

        if self.can_see(target) and dist <= self._shoot_range:
            # In sight and in range — stand still and engage
            self.set_walk()

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
            # Walk toward the target using the physics engine
            self._walk_toward(target)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _walk_toward(self, target) -> None:
        """Orient and set the forward-walk flag toward target."""
        self.look_toward(target)
        self.set_walk(up=True, sprint=True)

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
            self.add_bot(
                GuardBot.create(self, 'Guard-Blue', self.team_1, weapon)
            )
            self.add_bot(
                GuardBot.create(self, 'Guard-Green', self.team_2, weapon)
            )

    return GuardBotProtocol, connection
