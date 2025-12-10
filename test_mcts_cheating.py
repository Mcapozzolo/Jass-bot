# test_mcts_arena.py
#
# MonteCarloTrickAgent (cheating) gegen MinimaxTrickAgent (cheating).

import numpy as np

from jass.arena.arena import Arena
from jass.agents.agent_cheating_random_schieber import AgentCheatingRandomSchieber
from Minimax_Agent import MinimaxTrickAgent
from MCTS_Cheating import MonteCarloTrickAgent


def main():
    np.random.seed(0)

    nr_games_to_play = 100
    arena = Arena(nr_games_to_play, cheating_mode=True)

    mcts = MonteCarloTrickAgent(simulations_per_card=100)
    mini = MinimaxTrickAgent()
    rnd = AgentCheatingRandomSchieber()

    # Team 0: MCTS + Minimax
    # Team 1: zwei Random-Spieler  (oder was du willst)
    arena.set_players(mcts, rnd, mini, rnd)

    print(f"{nr_games_to_play} Spiele in der cheating-Arena (MCTS+Minimax vs Random)...")
    arena.play_all_games()

    p0 = arena.points_team_0.sum()
    p1 = arena.points_team_1.sum()

    print(f"Team 0 (MCTS+Minimax): {p0} Punkte")
    print(f"Team 1 (Random):        {p1} Punkte")
    print(f"Differenz:              {p0 - p1}")


if __name__ == "__main__":
    main()
