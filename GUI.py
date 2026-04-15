import tkinter as tk
import numpy as np
from pokemon import Pokemon, Move
from Battle_Sim import get_state
import neural_network

# --- Setup ---
tackle = Move("Tackle", "Normal", 40)
ember = Move("Ember", "Fire", 40)
water_gun = Move("Water Gun", "Water", 40)

p1 = Pokemon("Charmander", ["Fire"], 39, 52, 43, 65, [tackle, ember, tackle, tackle])
p2 = Pokemon("Squirtle", ["Water"], 44, 48, 65, 43, [tackle, water_gun, tackle, tackle])

nn = neural_network.simpleNN(6, 8, 4)

# --- UI ---
root = tk.Tk()
root.title("Pokemon AI Battle")
root.geometry("500x400")

enemy_label = tk.Label(root, text=f"{p2.name} HP: {p2.current_hp}")
enemy_label.pack()

log = tk.Text(root, height=10, width=50)
log.pack()

player_label = tk.Label(root, text=f"{p1.name} HP: {p1.current_hp}")
player_label.pack()

# --- Game Logic ---
def update_ui():
    player_label.config(text=f"{p1.name} HP: {p1.current_hp}")
    enemy_label.config(text=f"{p2.name} HP: {p2.current_hp}")

def ai_turn():
    if p2.current_hp <= 0:
        return
    
    state = np.array(get_state(p2, p1))
    outputs = nn.forward(state)
    move_index = np.argmax(outputs)
    move = p2.moves[move_index]

    dmg = p2.attack_target(p1, move)
    log.insert(tk.END, f"{p2.name} used {move.name} ({dmg} dmg)\n")

def player_move(index):
    if p1.current_hp <= 0 or p2.current_hp <= 0:
        return

    move = p1.moves[index]
    dmg = p1.attack_target(p2, move)
    log.insert(tk.END, f"{p1.name} used {move.name} ({dmg} dmg)\n")

    if p2.current_hp > 0:
        ai_turn()

    update_ui()
    log.see(tk.END)

def reset_game():
    p1.reset()
    p2.reset()
    log.delete("1.0", tk.END)
    update_ui()

# --- Buttons ---
for i in range(4):
    btn = tk.Button(root, text=p1.moves[i].name,
                    command=lambda i=i: player_move(i))
    btn.pack()

reset_btn = tk.Button(root, text="Reset", command=reset_game)
reset_btn.pack()

# --- Start ---
update_ui()
root.mainloop()