# minimax_agent.py
#
# Einfache Minimax-Implementierung für Jass im "cheating mode".
# Der Agent sieht das komplette GameState-Objekt und baut einen Minimax-Baum
# nur für den aktuellen Stich (4 Karten).

import numpy as np

from jass.agents.agent_cheating import AgentCheating
from jass.game.const import DIAMONDS, PUSH, team, card_strings
from jass.game.game_state import GameState
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber


class MinimaxTrickAgent(AgentCheating):
    """
    Cheating-Agent:
    - action_trump: sehr einfache, deterministische Trumpfwahl (egal fürs Minimax)
    - action_play_card: Minimax über den aktuellen Stich (4 Karten)
    """

    def __init__(self):
        self._rule = RuleSchieber()

    # ------------------------------------------------------------
    # Trumpfwahl (hier nur simpel, damit der Agent gültig spielt)
    # ------------------------------------------------------------
    def action_trump(self, state: GameState) -> int:
        """
        Sehr einfache Trumpf-Strategie:
        - Wenn wir Vorhand sind (forehand == -1), schieben wir einmal (PUSH).
        - Sonst wählen wir immer DIAMONDS als Trumpf.
        """
        # forehand == -1 bedeutet: dieser Spieler darf jetzt Trumpf wählen oder schieben
        if state.forehand == -1:
            return PUSH
        # Wenn schon geschoben wurde oder wir nicht Vorhand sind: einfach Trumpf 0 (DIAMONDS) nehmen
        return DIAMONDS

    # ------------------------------------------------------------
    # Minimax für aktuellen Stich
    # ------------------------------------------------------------
    def action_play_card(self, state: GameState) -> int:
        """
        Wählt eine Karte mit Minimax über den aktuellen Stich.

        Idee:
        - Wir kennen den kompletten GameState (cheating_mode).
        - Wir betrachten nur den laufenden Stich (bis 4 Karten).
        - Für jede mögliche Karte von uns:
            * Wir simulieren den Rest des Stiches komplett per Minimax.
            * Bewertung = Punkte dieses Stiches aus Sicht unseres Teams.
        - Maximier-Knoten: wenn Spieler aus unserem Team am Zug.
        - Minimier-Knoten: wenn Gegner am Zug.
        """
        # gültige Karten für den aktuellen Spieler
        valid_cards = self._rule.get_valid_cards_from_state(state)
        valid_indices = np.flatnonzero(valid_cards)

        if valid_indices.size == 0:
            # Sicherheits-Return, sollte nicht vorkommen
            return 0

        # Unser Team bestimmen (0 oder 1)
        my_player = state.player
        my_team = team[my_player]

        # Merken, welcher Stich gerade gespielt wird
        start_trick_index = state.nr_tricks

        best_card = int(valid_indices[0])
        best_value = -9999

        for card in valid_indices:
            # Kopie des Spielzustands über GameSim
            sim = GameSim(rule=self._rule)
            sim.init_from_state(state)

            # Wir spielen diese Karte
            sim.action_play_card(int(card))

            # Restlichen Stich per Minimax auswerten
            value = self._minimax_trick(sim, my_team, start_trick_index)

            if value > best_value:
                best_value = value
                best_card = int(card)

        # (Optional: Debug-Ausgabe, falls du schauen willst)
        # print(f"Minimax spielt: {card_strings[best_card]} mit Wert {best_value}")

        return best_card

    def _minimax_trick(self, sim: GameSim, my_team: int, start_trick_index: int) -> int:
        """
        Rekursive Minimax-Bewertung für den aktuellen Stich.

        Basisfall:
        - Wenn der Stich fertig ist (nr_tricks > start_trick_index),
          lesen wir trick_points und trick_winner und liefern
          +Punkte (wenn unser Team gewonnen hat) oder -Punkte (sonst).

        Rekursion:
        - Sonst: gültige Karten für aktuellen Spieler holen.
        - Wenn Spieler im my_team: max-Knoten.
        - Sonst: min-Knoten.
        """
        state = sim.state

        # Basisfall: der betrachtete Stich ist abgeschlossen
        if state.nr_tricks > start_trick_index:
            trick_idx = start_trick_index
            points = int(state.trick_points[trick_idx])
            winner = int(state.trick_winner[trick_idx])

            if team[winner] == my_team:
                return points
            else:
                return -points

        # Noch im laufenden Stich: nächster Spieler am Zug
        current_player = state.player
        valid_cards = self._rule.get_valid_cards_from_state(state)
        valid_indices = np.flatnonzero(valid_cards)

        if valid_indices.size == 0:
            # Sollte eigentlich nicht vorkommen
            return 0

        is_max_node = (team[current_player] == my_team)

        if is_max_node:
            best_value = -9999
            for card in valid_indices:
                child_sim = GameSim(rule=self._rule)
                child_sim.init_from_state(state)
                child_sim.action_play_card(int(card))

                value = self._minimax_trick(child_sim, my_team, start_trick_index)
                if value > best_value:
                    best_value = value
            return best_value
        else:
            best_value = 9999
            for card in valid_indices:
                child_sim = GameSim(rule=self._rule)
                child_sim.init_from_state(state)
                child_sim.action_play_card(int(card))

                value = self._minimax_trick(child_sim, my_team, start_trick_index)
                if value < best_value:
                    best_value = value
            return best_value
