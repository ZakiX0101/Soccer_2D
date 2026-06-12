import os
import sys

# Ajouter le chemin racine pour importer l'environnement
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from env.football_env import FootballEnv

def main():
    env = FootballEnv(render_mode="ansi")
    obs, info = env.reset()
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=== Football IA - Mode Joueur ===")
    print("Contrôles : Z (Haut), S (Bas), Q (Gauche), D (Droite) ou flèches correspondantes. 'quit' pour quitter.")
    print("But = G (Jaune/Zone), Agent = A, Défenseurs = D")
    print("-" * 40)
    
    env.render()
    
    done = False
    total_reward = 0.0
    
    # Mapping des touches aux actions
    key_mapping = {
        'z': 0, 'w': 0,  # UP
        's': 1,          # DOWN
        'q': 2, 'a': 2,  # LEFT
        'd': 3,          # RIGHT
    }
    
    while not done:
        user_input = input("Action (z/s/q/d) : ").strip().lower()
        
        if user_input in ('quit', 'exit'):
            print("Partie annulée.")
            break
            
        if user_input not in key_mapping:
            print("Touche invalide. Utilisez Z/S/Q/D.")
            continue
            
        action = key_mapping[user_input]
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        done = term or trunc
        
        # Effacer l'écran pour l'animation
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"Action : {user_input.upper()} | Récompense : {reward} | Score cumulé : {total_reward}")
        env.render()
        
        if done:
            print("-" * 40)
            if info.get("reached_goal"):
                print(f"🎉 GAGNÉ ! Vous avez atteint le but. Score final : {total_reward}")
            else:
                print(f"💀 PERDU ! Vous avez été capturé. Score final : {total_reward}")
            break

if __name__ == "__main__":
    main()
