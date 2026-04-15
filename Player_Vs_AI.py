import numpy as np
import neural_network
from pokemon import Pokemon, Move
from Battle_Sim import get_state
import copy

# ---------------- MOVES ----------------
Earthquake = Move("Earthquake", "Ground", 100)
Flamethrower = Move("Flamethrower", "Fire", 90)
Air_slash = Move("Air Slash", "Flying", 75)
Flare_Blitz = Move("Flare Blitz", "Fire", 120)

Surf = Move("Surf", "Water", 90)
Ice_Beam = Move("Ice Beam", "Ice", 90)
Dragon_Pulse = Move("Dragon Pulse", "Dragon", 90)

Thunderbolt = Move("Thunderbolt", "Electric", 90)
Volt_tackle = Move("Volt Tackle", "Electric", 120)
Play_Rough = Move("Play Rough", "Fairy", 85)

Sludge_Bomb = Move("Sludge Bomb", "Poison", 90)
Solar_Beam = Move("Solar Beam", "Grass", 100)
Giga_Drain = Move("Giga Drain", "Grass", 75)

# ---------------- POKEMON ----------------
Charizard = Pokemon("Charizard", ["Fire","Flying"], 78, 109, 85, 100,
                    [Earthquake, Flamethrower, Air_slash, Flare_Blitz])

Blastoise = Pokemon("Blastoise", ["Water"], 79, 85, 105, 78,
                    [Earthquake, Surf, Ice_Beam, Dragon_Pulse])

Pikachu = Pokemon("Pikachu", ["Electric"], 70, 100, 55, 90,
                  [Volt_tackle, Thunderbolt, Surf, Play_Rough])

Venusaur = Pokemon("Venusaur", ["Grass","Poison"], 80, 100, 100, 80,
                   [Sludge_Bomb, Solar_Beam, Giga_Drain, Earthquake])

all_pokemon = [Charizard, Blastoise, Pikachu, Venusaur]

# ---------------- LOAD AI ----------------
nn = neural_network.simpleNN(7, 8, 5)
nn.W1 = np.load("W1.npy")
nn.W2 = np.load("W2.npy")

# ---------------- PLAYER TEAM SELECT ----------------
print("Choose 2 Pokémon:")

for i, p in enumerate(all_pokemon):
    print(f"{i}: {p.name}")

choices = []
while len(choices) < 2:
    c = int(input("Enter choice: "))
    if c not in choices:
        choices.append(c)

team_player = [copy.deepcopy(all_pokemon[i]) for i in choices]

# ---------------- AI TEAM ----------------
team_ai = [copy.deepcopy(p) for p in np.random.choice(all_pokemon, 2, replace=False)]

print("\nAI Team:")
for p in team_ai:
    print(p.name)

# reset
for p in team_player + team_ai:
    p.reset()

active_player = 0
active_ai = 0

# ---------------- HELPERS ----------------
def print_moves(pokemon):
    print("\nMoves:")
    for i, move in enumerate(pokemon.moves):
        print(f"{i}: {move.name}")
    print("4: Switch")

def player_turn():
    global active_player

    p = team_player[active_player]
    e = team_ai[active_ai]

    print_moves(p)

    while True:
        try:
            choice = int(input("Enter move: "))
            if 0 <= choice <= 4:
                break
        except:
            pass

    if choice == 4:
        active_player = 1 - active_player
        print(f"You switched to {team_player[active_player].name}")
    else:
        p.attack_target(e, p.moves[choice])

def ai_turn():
    global active_ai

    p = team_ai[active_ai]
    e = team_player[active_player]

    state = np.array(get_state(p, e))
    outputs = nn.forward(state)
    action = np.argmax(outputs)

    if action == 4:
        active_ai = 1 - active_ai
        print(f"AI switched to {team_ai[active_ai].name}")
    else:
        move = p.moves[action]
        print(f"AI used {move.name}")
        p.attack_target(e, move)

# ---------------- GAME LOOP ----------------
while True:

    p1 = team_player[active_player]
    p2 = team_ai[active_ai]

    # auto switch if faint
    if p1.current_hp <= 0:
        active_player = 1 - active_player
        print(f"Your Pokémon fainted! Switched to {team_player[active_player].name}")
        continue

    if p2.current_hp <= 0:
        active_ai = 1 - active_ai
        print(f"AI switched to {team_ai[active_ai].name}")
        continue

    print(f"\n🔥 {p1.name} vs {p2.name}")

    if p1.speed >= p2.speed:
        player_turn()
        if p2.current_hp > 0:
            ai_turn()
    else:
        ai_turn()
        if p1.current_hp > 0:
            player_turn()

    # win check
    if all(p.current_hp <= 0 for p in team_player):
        print("\nAI wins!")
        break

    if all(p.current_hp <= 0 for p in team_ai):
        print("\nYou win!")
        break