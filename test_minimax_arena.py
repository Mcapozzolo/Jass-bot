# test_minimax_arena.py
#
# Testet MinimaxTrickAgent im cheating_mode der Arena.

import numpy as np

from jass.arena.arena import Arena
from jass.agents.agent_cheating_random_schieber import AgentCheatingRandomSchieber
from Minimax_Agent import MinimaxTrickAgent


def main():
    np.random.seed(0)

    # Arena im cheating_mode: Agent bekommt GameState statt GameObservation
    nr_games_to_play = 1000 

    arena = Arena(nr_games_to_play, cheating_mode=True)
    print("Arena erstellt")
    my = MinimaxTrickAgent()
    rnd = AgentCheatingRandomSchieber()

    # Team 0: Spieler 0 und 2 = Minimax
    # Team 1: Spieler 1 und 3 = Random
    arena.set_players(my, rnd, my, rnd)
    print("Spieler gesetzt, starte Spiele...")
    print(f"{nr_games_to_play} Spiele in der cheating-Arena mit Minimax...")

    arena.play_all_games()

    points_0 = arena.points_team_0.sum()
    points_1 = arena.points_team_1.sum()

    print(f"Team 0 (Minimax): {points_0} Punkte")
    print(f"Team 1 (Random):  {points_1} Punkte")
    print(f"Differenz:        {points_0 - points_1}")


if __name__ == "__main__":
    main()
