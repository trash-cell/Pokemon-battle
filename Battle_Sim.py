import numpy as np
from pokemon import type_chart, type_advantage
import random


def get_state(attacker, defender):
    """7-element state vector."""
    state = []
    state.append(attacker.current_hp / attacker.hp)
    state.append(defender.current_hp / defender.hp)
    for move in attacker.moves:
        mult = 1.0
        for t in defender.types:
            if move.type in type_chart and t in type_chart[move.type]:
                mult *= type_chart[move.type][t]
        state.append(mult)
    state.append(type_advantage(attacker, defender))
    return state


def calc_move_damages(attacker, defender):
    """
    Return expected damage for each move.
    Immune moves (0x) return -999 so they are never chosen.
    """
    damages = []
    for move in attacker.moves:
        mult = 1.0
        for t in defender.types:
            if move.type in type_chart and t in type_chart[move.type]:
                mult *= type_chart[move.type][t]
        if mult == 0.0:
            damages.append(-999.0)
        else:
            dmg = (attacker.attack / defender.defence) * move.bp * mult
            damages.append(dmg)
    return damages


def best_move(attacker, defender):
    """Return index of the move that deals the most damage."""
    return int(np.argmax(calc_move_damages(attacker, defender)))


def should_switch(attacker, bench, defender):
    """
    Pure rule: only switch if bench is alive AND strictly better typed
    than the current attacker against the defender.
    """
    if bench.current_hp <= 0:
        return False
    bench_score = 1.0
    for d_type in defender.types:
        for a_type in bench.types:
            if a_type in type_chart and d_type in type_chart[a_type]:
                bench_score *= type_chart[a_type][d_type]
    atk_score = 1.0
    for d_type in defender.types:
        for a_type in attacker.types:
            if a_type in type_chart and d_type in type_chart[a_type]:
                atk_score *= type_chart[a_type][d_type]
    return bench_score > 1.0 and bench_score > atk_score


def battle(team1, team2, nn1, nn2):
    """
    Hybrid battle:
    - Switch is handled by rule (should_switch), never by NN
    - NN picks a move (0-3), but best_move() is always computed
    - Reward = 1.0 if NN picked the best move, 0.0 otherwise
      (with a partial reward for picking a non-immune move)
    This teaches the NN to agree with the damage oracle over time.
    """
    active1 = 0
    active2 = 0

    states1, actions1, rewards1 = [], [], []
    states2, actions2, rewards2 = [], [], []

    MAX_TURNS = 200   # safety cap to prevent infinite loops
    turn = 0

    while turn < MAX_TURNS:
        turn += 1
        p1 = team1[active1]
        p2 = team2[active2]

        if p1.current_hp <= 0:
            ni = 1 - active1
            if team1[ni].current_hp > 0:
                active1 = ni
            continue
        if p2.current_hp <= 0:
            ni = 1 - active2
            if team2[ni].current_hp > 0:
                active2 = ni
            continue

        bench1 = team1[1 - active1]
        bench2 = team2[1 - active2]

        # ── rule-based switch check ────────────────────────────────────────────
        if should_switch(p1, bench1, p2):
            active1 = 1 - active1
            p1 = team1[active1]

        if should_switch(p2, bench2, p1):
            active2 = 1 - active2
            p2 = team2[active2]

        # ── NN picks move for p1 ──────────────────────────────────────────────
        state1  = np.array(get_state(p1, p2))
        out1    = nn1.forward(state1)
        best1   = best_move(p1, p2)
        damages1 = calc_move_damages(p1, p2)

        # 10% random exploration, otherwise NN argmax
        action1 = np.random.randint(4) if np.random.rand() < 0.1 else int(np.argmax(out1))

        # reward: 1.0 for picking the best move, 0.5 for picking any non-immune
        # move, 0.0 for picking an immune move
        if action1 == best1:
            r1 = 1.0
        elif damages1[action1] > 0:
            r1 = 0.5
        else:
            r1 = 0.0

        states1.append(state1)
        actions1.append(action1)
        rewards1.append(r1)

        # ── NN picks move for p2 ──────────────────────────────────────────────
        state2  = np.array(get_state(p2, p1))
        out2    = nn2.forward(state2)
        best2   = best_move(p2, p1)
        damages2 = calc_move_damages(p2, p1)

        action2 = np.random.randint(4) if np.random.rand() < 0.1 else int(np.argmax(out2))

        if action2 == best2:
            r2 = 1.0
        elif damages2[action2] > 0:
            r2 = 0.5
        else:
            r2 = 0.0

        states2.append(state2)
        actions2.append(action2)
        rewards2.append(r2)

        # ── execute BEST move always (not NN's choice) ────────────────────────
        # The NN learns which move is best, but the actual battle uses the
        # oracle so training battles are always high quality
        p1.attack_target(p2, p1.moves[best1])
        p2.attack_target(p1, p2.moves[best2])

        # ── win check ─────────────────────────────────────────────────────────
        if all(p.current_hp <= 0 for p in team1):
            return 2, states1, actions1, rewards1, states2, actions2, rewards2
        if all(p.current_hp <= 0 for p in team2):
            return 1, states1, actions1, rewards1, states2, actions2, rewards2

    # timed out
    return 0, states1, actions1, rewards1, states2, actions2, rewards2