# test_arena.py
import numpy as np

from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
from my_agent import MyAgent


def main():

    # Für reproduzierbare Ergebnisse
    np.random.seed(1)

    # Arena mit z.B. 200 Spielen
    numb=1000
    arena = Arena(nr_games_to_play=numb)
    print(numb, "Spiele in der Arena")
    # Team 0: Spieler 0 und 2 = dein Agent
    # Team 1: Spieler 1 und 3 = Random-Agent
    arena.set_players(
        MyAgent(),              # Spieler 0
        AgentRandomSchieber(),  # Spieler 1
        MyAgent(),              # Spieler 2
        AgentRandomSchieber()   # Spieler 3
    )

    # Alle Spiele durchsimulieren
    arena.play_all_games()
    points_0 = arena.points_team_0
    points_1 = arena.points_team_1

    total_0 = points_0.sum()
    total_1 = points_1.sum()
    diff    = total_0 - total_1
    n       = len(points_0)

    print(f"Anzahl Spiele:          {n}")
    print(f"Gesamtpunkte Team myagent:    {total_0}")
    print(f"Gesamtpunkte Team random:    {total_1}")
    print(f"Gesamtdifferenz (0-1):  {diff}")
    print()
    print(f"Ø Punkte pro Spiel Team myagent: {total_0 / n:.2f}")
    print(f"Ø Punkte pro Spiel Team random: {total_1 / n:.2f}")
    print(f"Ø Differenz pro Spiel:     {diff / n:.2f}")



if __name__ == "__main__":
    main()
