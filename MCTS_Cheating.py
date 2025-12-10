# MCTS_Cheating.py
#
# Einfache Monte-Carlo-Suche im cheating_mode:
# Der Agent sieht das komplette GameState-Objekt und bewertet
# jede gültige Karte durch mehrere zufällige Playouts bis zum Spielende.

import numpy as np

from jass.agents.agent_cheating import AgentCheating
from jass.game.const import DIAMONDS, team
from jass.game.game_state import GameState
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber


class MonteCarloTrickAgent(AgentCheating):
    """
    Cheating-Agent mit einfacher Monte-Carlo-Suche:

    - action_trump: wählt immer DIAMONDS (Trumpfwahl ist hier nicht der Fokus)
    - action_play_card: bewertet jede gültige Karte mit mehreren
      zufälligen Playouts bis zum Spielende und nimmt die Karte
      mit dem höchsten durchschnittlichen Punkte-Ergebnis für das eigene Team.
    """

    def __init__(self, simulations_per_card: int = 50):
        super().__init__()
        self._rule = RuleSchieber()
        self._sim = GameSim(self._rule)
        self._simulations_per_card = simulations_per_card
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------
    # Trumpfwahl (hier bewusst simpel)
    # ------------------------------------------------------------
    def action_trump(self, state: GameState) -> int:
        """
        Hier ist die Trumpfwahl nicht wichtig.
        Wir wählen einfach immer DIAMONDS.
        """
        return DIAMONDS

    # ------------------------------------------------------------
    # Hilfsfunktion: ein Playout für eine bestimmte Karte
    # ------------------------------------------------------------
    def _simulate_with_card(self, state: GameState, card: int, my_team: int) -> int:
        """
        Startet von 'state', spielt zuerst 'card' und simuliert
        dann den Rest des Spiels zufällig bis is_done() == True.

        Rückgabe:
            Endpunkte meines Teams (0 oder 1) im simulierten Spiel.
        """
        # Startzustand kopieren
        self._sim.init_from_state(state)

        # Unsere Karte spielen
        self._sim.action_play_card(card)

        # Rest des Spiels zufällig fertig spielen
        while not self._sim.is_done():
            sim_state = self._sim.state
            valid_cards = self._rule.get_valid_cards_from_state(sim_state)
            valid_indices = np.flatnonzero(valid_cards)

            if len(valid_indices) == 0:
                # sollte eigentlich nie passieren, Sicherheitsnetz
                break

            next_card = self._rng.choice(valid_indices)
            self._sim.action_play_card(next_card)

        # Am Ende: Punkte meines Teams
        return int(self._sim.state.points[my_team])

    # ------------------------------------------------------------
    # Monte-Carlo Kartenwahl
    # ------------------------------------------------------------
    def action_play_card(self, state: GameState) -> int:
        """
        Wählt eine Karte über Monte-Carlo-Playouts:

        - Bestimme alle gültigen Karten
        - Für jede Karte führe N Simulationen bis zum Spielende durch
        - Wert = durchschnittliche Endpunkte meines Teams
        - Nimm die Karte mit dem höchsten Durchschnittswert
        """
        # Gültige Karten für den aktuellen Spieler
        valid_cards = self._rule.get_valid_cards_from_state(state)
        valid_indices = np.flatnonzero(valid_cards)

        # Wenn nur eine Karte möglich ist, direkt spielen
        if len(valid_indices) == 1:
            return int(valid_indices[0])

        current_player = state.player
        my_team = team[current_player]

        best_card = None
        best_value = -1.0

        # Jede gültige Karte durchprobieren
        for card in valid_indices:
            total_score = 0

            # Mehrere zufällige Playouts für diese Karte
            for _ in range(self._simulations_per_card):
                total_score += self._simulate_with_card(state, int(card), my_team)

            avg_score = total_score / self._simulations_per_card

            if avg_score > best_value:
                best_value = avg_score
                best_card = int(card)

        return best_card
