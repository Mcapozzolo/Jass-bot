import numpy as np
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.agents.agent import Agent

# ---------------------------------------------------------------------------
# Score-Tabellen wie im Notebook (Task 2)
# Index 0..8 entspricht: Ass, König, Dame, Bauer, 10, 9, 8, 7, 6
# ---------------------------------------------------------------------------

TRUMP_SCORE = [15, 10, 7, 25, 6, 19, 5, 5, 5]
NO_TRUMP_SCORE = [9, 7, 5, 2, 1, 0, 0, 0, 0]
OBENABE_SCORE = [14, 10, 8, 7, 5, 0, 5, 0, 0]
UNE_UFE_SCORE = [0, 2, 1, 1, 5, 5, 7, 9, 11]


def calculate_trump_selection_score(cards, trump: int) -> int:
    """
    Berechnet den Score für eine Hand (int-encodierte Kartenliste) und einen gegebenen Trumpf.
    Entspricht der Implementierung aus Task 2.
    """
    score = 0

    for card in cards:
        c = color_of_card[card]   # 0..3
        r = offset_of_card[card]  # 0..8 (Ass..6)

        # obenabe / uneufe: nur Rang zählt
        if trump == OBE_ABE:
            score += OBENABE_SCORE[r]
        elif trump == UNE_UFE:
            score += UNE_UFE_SCORE[r]
        else:
            # normaler Trumpf
            if c == trump:
                score += TRUMP_SCORE[r]
            else:
                score += NO_TRUMP_SCORE[r]

    return score


class MyAgent(Agent):
    """
    Dein Jass-Agent:
    - wählt Trumpf mit der Score-Heuristik
    - spielt zufällige gültige Karten
    """

    def __init__(self):
        super().__init__()
        self._rule = RuleSchieber()

    def action_trump(self, obs) -> int:
        """
        Entscheidet, welchen Trumpf zu wählen (oder ob geschoben wird).
        obs: GameObservation (vom jass-kit).
        """
        # Handkarten als int-Liste 0..35
        card_list = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)

        possible_trumps = [CLUBS, SPADES, HEARTS, DIAMONDS, OBE_ABE, UNE_UFE]

        best_trump = None
        best_score = -9999

        for t in possible_trumps:
            s = calculate_trump_selection_score(card_list, t)
            if s > best_score:
                best_score = s
                best_trump = t

        # Schwellwert wie im Script: unter 68 → schieben, falls erlaubt
        THRESHOLD = 68

        # push_allowed ist evtl. nicht immer gesetzt → getattr mit Default False
        can_push = getattr(obs, "push_allowed", False)

        if best_score < THRESHOLD and can_push:
            return PUSH

        return best_trump

    def action_play_card(self, obs) -> int:

      

        return self.play_worst_card(obs)
    

    def play_worst_card(self, obs) -> int:
        """
        Wählt die schlechteste Karte, die gespielt wird (für Debugging-Zwecke).
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_indices = np.flatnonzero(valid_cards)

        worst_card = None
        worst_trumpcard = None
        worst_score = -1
        worst_trumpscore = -1


        non_trumps = []
        for card in valid_indices:
            if color_of_card[card] != obs.trump:
                non_trumps.append(card)

        if len(non_trumps) > 0:
            for card in non_trumps:
                r = offset_of_card[card]
                if r > worst_score:
                    worst_score = r
                    worst_card = int(card)
        else:
            for card in valid_indices:
                r = offset_of_card[card]
                if r > worst_score:
                    worst_score = r
                    worst_card = int(card)
        return worst_card