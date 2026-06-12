import os
import sys
import pygame
import numpy as np

# Ajouter le chemin racine
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from env.football_env import FootballEnv
from agents.genetic_agent import GeneticAgent

# --- Configuration Pygame ---
pygame.init()

CELL_SIZE = 60
GRID_SIZE = 10
HEADER_HEIGHT = 80
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE + HEADER_HEIGHT

# Couleurs
COLOR_BG = (30, 30, 30)
COLOR_GRASS_1 = (34, 139, 34)
COLOR_GRASS_2 = (40, 150, 40)
COLOR_TEXT = (255, 255, 255)
COLOR_AGENT = (30, 144, 255)       # Bleu
COLOR_DEFENDER = (220, 20, 60)     # Rouge
COLOR_GOAL = (255, 215, 0)         # Jaune Or

# Polices (sécurisé avec fallback)
try:
    font_title = pygame.font.SysFont("Segoe UI", 32, bold=True)
    font_info = pygame.font.SysFont("Segoe UI", 24)
    font_small = pygame.font.SysFont("Segoe UI", 18)
except:
    font_title = pygame.font.Font(None, 40)
    font_info = pygame.font.Font(None, 30)
    font_small = pygame.font.Font(None, 24)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Football 2D - Simulation IA")
clock = pygame.time.Clock()

def draw_grid(screen):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            color = COLOR_GRASS_1 if (r + c) % 2 == 0 else COLOR_GRASS_2
            rect = pygame.Rect(c * CELL_SIZE, HEADER_HEIGHT + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)

def draw_entity(screen, pos, color, radius_ratio=0.35, outline=True):
    r, c = pos
    center_x = c * CELL_SIZE + CELL_SIZE // 2
    center_y = HEADER_HEIGHT + r * CELL_SIZE + CELL_SIZE // 2
    radius = int(CELL_SIZE * radius_ratio)
    pygame.draw.circle(screen, color, (center_x, center_y), radius)
    if outline:
        pygame.draw.circle(screen, (255, 255, 255), (center_x, center_y), radius, 2)

def main():
    env = FootballEnv()
    
    # Charger l'agent IA
    genome_path = os.path.join(ROOT, "results", "best_genome.npy")
    ai_agent = None
    if os.path.exists(genome_path):
        genome = np.load(genome_path)
        ai_agent = GeneticAgent(genome)
    
    state = "MENU" # MENU, PLAYING, GAME_OVER
    mode = None    # "MANUAL", "AI"
    obs, info = env.reset()
    total_reward = 0.0
    
    running = True
    ai_timer = 0
    AI_DELAY = 300 # millisecondes entre chaque pas de l'IA
    
    message = ""
    
    while running:
        dt = clock.tick(60) # 60 FPS
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if state == "MENU":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        mode = "MANUAL"
                        state = "PLAYING"
                        obs, info = env.reset()
                        total_reward = 0.0
                        message = ""
                    elif event.key == pygame.K_a and ai_agent is not None:
                        mode = "AI"
                        state = "PLAYING"
                        obs, info = env.reset()
                        total_reward = 0.0
                        ai_timer = pygame.time.get_ticks()
                        message = ""
            
            elif state == "PLAYING" and mode == "MANUAL":
                if event.type == pygame.KEYDOWN:
                    action = None
                    if event.key in (pygame.K_z, pygame.K_w, pygame.K_UP):
                        action = 0
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        action = 1
                    elif event.key in (pygame.K_q, pygame.K_a, pygame.K_LEFT):
                        action = 2
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        action = 3
                        
                    if action is not None:
                        obs, reward, term, trunc, info = env.step(action)
                        total_reward += reward
                        if term or trunc:
                            state = "GAME_OVER"
                            if info.get("reached_goal"):
                                message = "🎉 GAGNÉ !"
                            else:
                                message = "💀 PERDU !"

            elif state == "GAME_OVER":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        state = "MENU"
                        message = ""

        # --- Logique IA ---
        if state == "PLAYING" and mode == "AI":
            now = pygame.time.get_ticks()
            if now - ai_timer > AI_DELAY:
                ai_timer = now
                action = ai_agent.select_action(obs)
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                if term or trunc:
                    state = "GAME_OVER"
                    if info.get("reached_goal"):
                        message = "🎉 GAGNÉ !"
                    else:
                        message = "💀 PERDU !"
        
        # --- Dessin ---
        screen.fill(COLOR_BG)
        
        if state == "MENU":
            title = font_title.render("Football 2D IA", True, COLOR_TEXT)
            screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//3 - 30))
            
            txt_m = font_info.render("[M] Jouer Manuellement", True, COLOR_TEXT)
            screen.blit(txt_m, (WIDTH//2 - txt_m.get_width()//2, HEIGHT//2))
            
            if ai_agent is not None:
                txt_a = font_info.render("[A] Lancer l'IA (GeneticAgent)", True, COLOR_TEXT)
                screen.blit(txt_a, (WIDTH//2 - txt_a.get_width()//2, HEIGHT//2 + 40))
            else:
                txt_a = font_small.render("(IA non disponible - lancez l'entraînement d'abord)", True, (150, 150, 150))
                screen.blit(txt_a, (WIDTH//2 - txt_a.get_width()//2, HEIGHT//2 + 40))
                
        else:
            # Header
            score_txt = font_info.render(f"Score: {total_reward:.1f} | Mode: {mode}", True, COLOR_TEXT)
            screen.blit(score_txt, (10, 10))
            
            if state == "GAME_OVER":
                msg_color = (100, 255, 100) if "GAGNÉ" in message else (255, 100, 100)
                msg_txt = font_title.render(f"{message} [R pour Menu]", True, msg_color)
                screen.blit(msg_txt, (WIDTH//2 - msg_txt.get_width()//2, 40))
            else:
                controls_txt = font_small.render("Z/Q/S/D ou Flèches pour bouger" if mode == "MANUAL" else "L'IA réfléchit...", True, (200, 200, 200))
                screen.blit(controls_txt, (10, 45))
            
            # Terrain
            draw_grid(screen)
            
            # Entités
            draw_entity(screen, env._goal_pos, COLOR_GOAL, radius_ratio=0.4, outline=True)
            for d_pos in env._defender_pos:
                draw_entity(screen, d_pos, COLOR_DEFENDER, radius_ratio=0.35, outline=True)
            draw_entity(screen, env._agent_pos, COLOR_AGENT, radius_ratio=0.35, outline=True)
            
        pygame.display.flip()
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
