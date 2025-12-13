import os
import numpy as np
import joblib

from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.game_sim import GameSim
from jass.game.game_state_util import state_from_observation
from jass.agents.agent import Agent


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


class MyAgentcomplex(Agent):
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
        
        # RNG für MCTS
        self._rng = np.random.default_rng()

        # MCTS-Parameter
        self._mcts_iterations = 200
        self._mcts_exploration_c = 1.4

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
        Wählt eine gültige Karte mit Monte Carlo Tree Search (Root-MCTS + Determinization).

        - Knoten = aktueller Zustand (obs)
        - Kanten = mögliche Karten
        - UCB1 steuert Exploration vs. Exploitation
        """
        valid_mask = self._rule.get_valid_cards_from_obs(obs)
        valid_cards = np.flatnonzero(valid_mask)

        # Trivialfall: nur eine gültige Karte
        if valid_cards.size == 1:
            return int(valid_cards[0])

        my_player = obs.player
        my_team = 0 if my_player in (0, 2) else 1

        # Statistik pro Karte (global über alle Determinizations)
        N = np.zeros(36, dtype=np.int32)    # Besuchszahlen
        W = np.zeros(36, dtype=np.float32)  # Summe der Rewards

        C = self._mcts_exploration_c
        iterations = self._mcts_iterations

        for it in range(iterations):
            # ---- 1) Determinization: versteckte Hände sampeln ----
            hands = self._sample_hidden_hands(obs)

            # ---- 2) State aus Observation + Händen erzeugen ----
            state = state_from_observation(obs, hands)

            # ---- 3) Selection: wähle Karte mit UCB1 ----
            total_visits = 1 + N[valid_cards].sum()

            best_ucb = -1e18
            best_card = int(valid_cards[0])

            for card in valid_cards:
                n = N[card]
                if n == 0:
                    ucb = 1e9  # Erzwinge mind. 1 Besuch
                else:
                    exploit = W[card] / n
                    explore = C * np.sqrt(np.log(total_visits) / n)
                    ucb = exploit + explore

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_card = int(card)

            # ---- 4) Expansion + Simulation: spiele best_card & rollout ----
            # Wir kopieren den State in einen GameSim und führen best_card aus
            sim = GameSim(rule=self._rule)
            sim.init_from_state(state)

            sim.action_play_card(best_card)

            # Rest zufällig spielen
            reward = self._simulate_random_game(sim.state, my_team)

            # ---- 5) Backpropagation: Statistik updaten ----
            N[best_card] += 1
            W[best_card] += reward

        # ---- 6) Aktion wählen: Karte mit den meisten Besuchen ----
        visits_valid = N[valid_cards]
        best_idx = int(np.argmax(visits_valid))
        best_card_final = int(valid_cards[best_idx])

        return best_card_final

        
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

    def _simulate_random_game(self, state, my_team: int) -> float:
        """
        Rollout: spiele von diesem State aus zufällig zu Ende und
        gib (Punkte_mein_Team - Punkte_anderes_Team) zurück.
        """
        sim = GameSim(rule=self._rule)
        sim.init_from_state(state)

        while not sim.is_done():
            current_player = sim.state.player
            valid = self._rule.get_valid_cards_from_state(sim.state)
            valid_indices = np.flatnonzero(valid)
            if valid_indices.size == 0:
                break
            card = int(self._rng.choice(valid_indices))
            sim.action_play_card(card)

        # Punkte auslesen
        points0 = int(sim.state.points[0])
        points1 = int(sim.state.points[1])

        if my_team == 0:
            return float(points0 - points1)
        else:
            return float(points1 - points0)
