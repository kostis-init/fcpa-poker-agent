from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel
import re
from format_mappings import encode_hand
import play_fcpa

map_action_to_int = {
    '': -1,  # new action is first in game
    '|': -1,  # new action is first in round
    'f': 0,
    'c': 1,
    'p': 2,
    'a': 3,
    # TODO check if not better to put raise and value as input in neural instead of 'c' and 'a'
}


def get_encoded_cards_from_state(state) -> ([float], [float]):
    encoded_cards1 = encode_hand(get_cards_from_state(state))
    return encoded_cards1


def get_cards_from_state(state) -> [str]:
    result = re.search('Public: (.*)\]\[', str(state))
    s = result.group(1)
    publicCards = np.array([s[i:i + 2] for i in range(0, len(s), 2)], dtype=object)

    result = re.search('Private: (.*)\]\[Pub', str(state))
    s = result.group(1)
    privateCards = np.array([s[i:i + 2] for i in range(0, len(s), 2)], dtype=object)

    return np.concatenate((privateCards, publicCards))


def get_string_from_state(state, description) -> str:
    result = re.findall(description + ': (.*?)]', str(state))
    if result == []:
        return ''
    return result[0]


def get_actions_from_state(state) -> str:
    return get_string_from_state(state, 'Sequences')


def get_last_player_id(state) -> int:
    return int(get_string_from_state(state, 'Player'))


def get_other_player(player_id: int) -> int:
    if player_id == 1:
        return 0
    else:
        return 1


def get_last_action_from_state(state) -> int:
    actions_str = get_string_from_state(state, 'Sequences')
    if actions_str == '':  # first move
        return map_action_to_int['']
    elif len(actions_str) == 1:  # c (check/call)
        return map_action_to_int[actions_str[-1]]
    else:  # bet/raise pot OR allin
        player_id = get_other_player(get_last_player_id(state))
        if get_wallets_from_state(state)[player_id] == 0:  # bet everything so means it was allin
            return map_action_to_int['a']
        return map_action_to_int['c']


def get_pot_from_state(state) -> int:
    return int(get_string_from_state(state, 'Pot'))


def get_wallets_from_state(state) -> (int, int):
    stringMoney = get_string_from_state(state, "Money")
    str1, str2 = stringMoney.split(' ')
    return int(str1), int(str2)


def get_round_from_state(state: str) -> int:
    return int(re.findall('Round (.*?)]', str(state))[0])


def main(_):
    game, agents, rng = play_fcpa.create_fcpa_with_agents("random", "random")

    states, utilities, historyActions = play_fcpa.generate_states(game, agents, rng)
    print("historyActions", historyActions, "states", states)
    states, utilities, historyActions = play_fcpa.generate_states(game, agents, rng)
    print("historyActions", historyActions, "states", states)

    print("utilities:", utilities)
    for state in states:
        print("String state:", state)
        print("round:", get_round_from_state(state))
        print("pot:", get_pot_from_state(state))
        print("wallet money:", get_wallets_from_state(state))
        print("all actions string:", get_actions_from_state(state))
        print("last actions:", get_last_action_from_state(state))

    return


if __name__ == "__main__":
    app.run(main)
