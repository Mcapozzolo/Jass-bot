# extract_trump_data.py
#
# Extrahiert Trumpf-Trainingsdaten aus den Swisslos-Logs
# jass_game_000x.txt
#
# Output: Data/trump_train_sw.npz mit X (N x 36), y (N)

import json
import glob
import numpy as np

from jass.game.const import card_strings

# Mapping Kartenstring -> Index 0..35
CARD_INDEX = {s: i for i, s in enumerate(card_strings)}


def game_to_example(line: str):
    """
    Nimmt eine JSON-Zeile (ein Spiel) und liefert (x, y):
      x: 36-dim One-Hot-Hand des Trumpfspielers
      y: trump (int)
    oder None, wenn etwas nicht passt.
    """
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return None

    if "game" not in entry:
        return None

    game = entry["game"]

    # Trumpf-Label
    if "trump" not in game:
        return None
    trump = int(game["trump"])  # 0..5

    dealer = int(game.get("dealer", 0))
    forehand = int(game.get("forehand", (dealer + 1) % 4))
    tricks = game.get("tricks", [])

    # Hände rekonstruieren: jede gespielte Karte gehört zum Spieler,
    # der an dieser Stelle in der Stichreihenfolge ist.
    hands = {0: [], 1: [], 2: [], 3: []}

    for trick in tricks:
        cards = trick.get("cards", [])
        first = int(trick.get("first", 0))
        if len(cards) != 4:
            return None

        for i, card_str in enumerate(cards):
            player = (first + i) % 4
            hands[player].append(card_str)

    # Jede Hand sollte 9 Karten haben
    if any(len(hands[p]) != 9 for p in range(4)):
        return None

    trump_player = forehand
    hand_cards = hands[trump_player]

    # 36-dim One-Hot
    x = np.zeros(36, dtype=np.int8)
    for cs in hand_cards:
        idx = CARD_INDEX.get(cs)
        if idx is None:
            return None
        x[idx] = 1

    y = trump
    return x, y


def main():
    # Passe das Muster an deinen Pfad an:
    file_pattern = "Data/jass_game_*.txt"
    files = sorted(glob.glob(file_pattern))

    if not files:
        print(f"Keine Dateien gefunden für Pattern: {file_pattern}")
        return

    X_list = []
    Y_list = []

    for path in files:
        print(f"Verarbeite: {path}")
        with open(path, "r") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                result = game_to_example(line)
                if result is None:
                    continue
                x, y = result
                X_list.append(x)
                Y_list.append(y)

    if not X_list:
        print("Keine Beispiele extrahiert.")
        return

    X = np.stack(X_list, axis=0)
    y = np.array(Y_list, dtype=np.int8)

    print("Gesammelte Beispiele:", X.shape[0])
    print("Feature-Dimension:", X.shape[1])

    # Speichern als .npz
    out_path = "Data/trump_train_sw.npz"
    np.savez(out_path, X=X, y=y)
    print(f"Gespeichert nach: {out_path}")


if __name__ == "__main__":
    main()
