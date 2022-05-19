"""Python spiel example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from open_spiel.python.bots.policy import PolicyBot

import sys
import argparse
import logging
import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots

from nn_confidence_level import loadNN_confidence_level
from fcpaPolicy import FcpaPolicy
from nn_actions import load_nn

logger = logging.getLogger('be.kuleuven.cs.dtai.fcpa')


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.
    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    rng = np.random.RandomState()

    games_list = pyspiel.registered_names()
    assert "universal_poker" in games_list
    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    print("Creating game: {}".format(fcpa_game_string))
    game = pyspiel.load_game(fcpa_game_string)

    nn_actions = load_nn()
    nn_confidence_level = loadNN_confidence_level()

    policy_fcpa = FcpaPolicy(game, [0, 1], nn_actions, nn_confidence_level)
    my_player = PolicyBot(player_id, rng, policy_fcpa)

    return my_player


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
    game = pyspiel.load_game(fcpa_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0, 1]]
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())
