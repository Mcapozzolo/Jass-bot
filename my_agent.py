import os
import numpy as np
import joblib

from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.agents.agent import Agent

# ---------------------------------------------------------
# Einfache Bewertungs-Tabellen pro Rang (0..8)
# Rang 0 = Ass, 1 = König, 2 = Dame, 3 = Bauer, 4 = 10, 5 = 9, 6 = 8, 7 = 7, 8 = 6
# Diese Tabellen kommen aus dem Notebook / der Übung.
# ---------------------------------------------------------

TRUMP_SCORE = [15, 10, 7, 25, 6, 19, 5, 5, 5]
NO_TRUMP_SCORE = [9, 7, 5, 2, 1, 0, 0, 0, 0]
OBENABE_SCORE = [14, 10, 8, 7, 5, 0, 5, 0, 0]
UNE_UFE_SCORE = [0, 2, 1, 1, 5, 5, 7, 9, 11]


def score_hand_for_trump(card_list, trump) -> int:
    """
    Bewertet eine Hand (Liste von int-Karten 0..35) für einen gegebenen Trumpf.
    Wird nur für die heuristische Trumpf-Alternative verwendet.
    """
    score = 0
    for card in card_list:
        color = color_of_card[card]   # 0..3
        rank = offset_of_card[card]   # 0..8

        if trump == OBE_ABE:
            score += OBENABE_SCORE[rank]
        elif trump == UNE_UFE:
            score += UNE_UFE_SCORE[rank]
        else:
            if color == trump:
                score += TRUMP_SCORE[rank]
            else:
                score += NO_TRUMP_SCORE[rank]

    return score


def card_strength(card: int, trump: int) -> int:
    """
    Sehr einfache Heuristik für die Stärke einer Karte.
    Höherer Wert = stärkere Karte.
    """
    color = color_of_card[card]
    rank = offset_of_card[card]

    if trump == OBE_ABE:
        return OBENABE_SCORE[rank]
    if trump == UNE_UFE:
        return UNE_UFE_SCORE[rank]

    if color == trump:
        return TRUMP_SCORE[rank]
    else:
        return NO_TRUMP_SCORE[rank]


class MyAgent(Agent):
    """
    Mein Jass-Agent:
    - Trumpfwahl mit ML-Modell (Multi-Layer-Perceptron aus scikit-learn)
      * Input: Hand als One-Hot-Vektor (1x36)
      * Output: Klasse 0..5 (CLUBS, SPADES, HEARTS, DIAMONDS, OBE_ABE, UNE_UFE)
      * Wenn Modell unsicher ist und Schieben erlaubt ist → PUSH.
      * Wenn Modell nicht geladen werden kann → einfache Heuristik.
    - Kartenwahl:
      * In der frühen Phase (mehr als 5 Karten): schwache Karte abwerfen,
        möglichst keine Trumpfkarte.
      * In der späten Phase (5 oder weniger Karten): starke Karte spielen.
    """

    def __init__(self):
        super().__init__()
        self._rule = RuleSchieber()

        # Trumpf-ML-Modell laden
        model_path = os.path.join(os.path.dirname(__file__), 'Data', 'trump_model_sw.joblib')
        try:
            self._trump_model = joblib.load(model_path)
            print(f"[MyAgent] Trumpfmodell geladen: {model_path}")
        except Exception as e:
            print(f"[MyAgent] Konnte Trumpfmodell nicht laden ({model_path}): {e}")
            self._trump_model = None

    # ---------------------------------------------------------
    # Trumpfwahl
    # ---------------------------------------------------------
    def action_trump(self, obs) -> int:
        """
        Wählt den Trumpf:
        1. Wenn ML-Modell vorhanden → Modellvorhersage + Unsicherheitscheck.
        2. Sonst → heuristische Bewertung der Hand.
        """

        # Hand als 1x36-Featurevektor
        hand_vec = np.array(obs.hand, dtype=np.float32).reshape(1, -1)

        # --- 1) Versuch: ML-Modell ---
        if self._trump_model is not None:
            # Wahrscheinlichkeiten für jede Trumpf-Klasse
            proba = self._trump_model.predict_proba(hand_vec)[0]
            best_class = int(np.argmax(proba))
            best_conf = float(proba[best_class])

            # Darf ich schieben?
            can_push = getattr(obs, "push_allowed", False)
            CONF_THRESHOLD = 0.30  # falls Modell weniger als 30% sicher ist

            if can_push and best_conf < CONF_THRESHOLD:
                return PUSH

            return best_class  # 0..5 →

        # --- 2) Fallback: Heuristik ---
        card_list = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        possible_trumps = [CLUBS, SPADES, HEARTS, DIAMONDS, OBE_ABE, UNE_UFE]

        best_trump = None
        best_score = -999999

        for t in possible_trumps:
            s = score_hand_for_trump(card_list, t)
            if s > best_score:
                best_score = s
                best_trump = t

        # Einfache Push-Logik wie im Notebook
        THRESHOLD = 68
        can_push = getattr(obs, "push_allowed", False)
        if can_push and best_score < THRESHOLD:
            return PUSH

        return best_trump

    # ---------------------------------------------------------
    # Kartenwahl
    # ---------------------------------------------------------
    def action_play_card(self, obs) -> int:
        """
        Wählt eine gültige Karte aus obs.hand.

        Idee:
        - Wenn noch viele Karten ( > 5 ) auf der Hand sind:
            * schwächste Karte spielen, bevorzugt keine Trumpfkarte
        - Wenn nur noch wenige Karten ( <= 5 ):
            * stärkste Karte spielen
        """
        valid_mask = self._rule.get_valid_cards_from_obs(obs)
        valid_indices = np.flatnonzero(valid_mask)

        # Sicherheitscheck (sollte nie passieren)
        if len(valid_indices) == 0:
            return 0

        trump = obs.trump
        cards_left = int(np.sum(obs.hand))

        # Frühe Phase: viele Karten → schlechte Karte loswerden
        if cards_left > 5:
            non_trump_cards = []

            # Nur relevant, wenn ein Farb-Trumpf aktiv ist
            if trump in [CLUBS, SPADES, HEARTS, DIAMONDS]:
                for card in valid_indices:
                    if color_of_card[card] != trump:
                        non_trump_cards.append(card)

            # Wenn es non-trump-Karten gibt, nehmen wir diese als Kandidaten,
            # sonst müssen wir aus allen gültigen Karten wählen.
            if len(non_trump_cards) > 0:
                candidates = non_trump_cards
            else:
                candidates = list(valid_indices)

            worst_card = candidates[0]
            worst_value = card_strength(worst_card, trump)

            for card in candidates[1:]:
                value = card_strength(card, trump)
                if value < worst_value:
                    worst_value = value
                    worst_card = card

            return int(worst_card)

        # Späte Phase: wenige Karten → starke Karte spielen
        else:
            best_card = valid_indices[0]
            best_value = card_strength(best_card, trump)

            for card in valid_indices[1:]:
                value = card_strength(card, trump)
                if value > best_value:
                    best_value = value
                    best_card = card

            return int(best_card)
        
    def _sample_hidden_hands(self, obs) -> np.ndarray:
        """
        Erzeuge eine zufällige, konsistente Verteilung der unbekannten Karten
        auf die Gegnerhände.

        Rückgabe:
            hands: Array der Form (4, 36), one-hot, inkl. unserer eigenen Hand.
        """
        hands = np.zeros((4, 36), dtype=np.int32)

        me = obs.player        # aktueller Spielerindex (0..3)
        hands[me, :] = obs.hand

        # --- 1) Gespielte Karten finden ------------------------------------
        played_mask = np.zeros(36, dtype=bool)
        cards_played_by_player = np.zeros(4, dtype=np.int32)

        # Vollständig abgeschlossene Stiche
        for t in range(obs.nr_tricks):
            first = obs.trick_first_player[t]
            if first == -1:
                continue
            for pos in range(4):
                card = int(obs.tricks[t, pos])
                if card == -1:
                    continue
                played_mask[card] = True
                player = (first + pos) & 3
                cards_played_by_player[player] += 1

        # Aktueller (noch nicht vollständiger) Stich
        if obs.nr_cards_in_trick > 0:
            first = obs.trick_first_player[obs.nr_tricks]
            if first != -1:
                for pos in range(obs.nr_cards_in_trick):
                    card = int(obs.current_trick[pos])
                    if card == -1:
                        continue
                    played_mask[card] = True
                    player = (first + pos) & 3
                    cards_played_by_player[player] += 1

        # --- 2) Wie viele Karten sollte jeder Spieler noch haben? ----------
        remaining_per_player = 9 - cards_played_by_player
        # unsere Hand kennen wir genau:
        remaining_per_player[me] = int(np.sum(obs.hand))

        # --- 3) Unbekannte Karten einsammeln -------------------------------
        unknown_cards = [
            c for c in range(36)
            if (not played_mask[c]) and (obs.hand[c] == 0)
        ]

        self._rng.shuffle(unknown_cards)

        others = [p for p in range(4) if p != me]

        # --- 4) Unbekannte Karten auf Gegner verteilen ---------------------
        idx = 0
        for p in others:
            need = int(remaining_per_player[p])
            cards_for_p = unknown_cards[idx:idx + need]
            for card in cards_for_p:
                hands[p, card] = 1
            idx += need

        return hands

