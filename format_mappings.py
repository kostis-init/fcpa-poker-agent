from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
import numpy as np
import combinations

mapValToInt = {
    '2': 0,
    '3': 1,
    '4': 2,
    '5': 3,
    '6': 4,
    '7': 5,
    '8': 6,
    '9': 7,
    'T': 8,
    'J': 9,
    'Q': 10,
    'K': 11,
    'A': 12,
}

mapColorToInt = {
    'c': 0,
    'd': 1,
    'h': 2,
    's': 3
}


def cardString_to_pair_float(cardStr: str):
    val = mapValToInt[cardStr[0]]
    color = mapColorToInt[cardStr[1]]

    return float(val), float(color)


def cardString_to_int(cardStr: str) -> int:
    val = mapValToInt[cardStr[0]]
    color = mapColorToInt[cardStr[1]]

    return val + color * 12


"""deprecated"""


def listcards_to_liststring(hands):
    # return  return np.array([cardString_to_int(xi) for xi in x],dtype=int)
    print(hands.shape)
    int_hands = np.zeros(hands.shape, dtype=object)
    for i_hand in range(len(hands)):
        hand = []
        for card in hands[i_hand]:
            hand.append(cardString_to_int(card))
        int_hands[i_hand] = hand
    return int_hands


def encode_hand(hand):
    hand_int = []
    for card in hand:
        val, color = cardString_to_pair_float(card)
        # hand_int.append(cardString_to_int(card))
        hand_int.append(val)
        hand_int.append(color)
    hand_int = combinations.get_combinations_of_cards(hand_int)
    return hand_int


if __name__ == "__main__":
    app.run()
