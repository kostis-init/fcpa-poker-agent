#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python spiel example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel
from open_spiel.python.bots.policy import PolicyBot
import fcpaNeverFoldPolicy

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 12761381, "The seed to use for the RNG.")

# Supported types of players: "random", "human", "check_call", "fold", "50/50_call-fold"
flags.DEFINE_string("player0", "check_call", "Type of the agent for player 0.")
flags.DEFINE_string("player1", "check_call", "Type of the agent for player 1.")


def LoadAgent(agent_type, game, player_id, rng):
    """Return a bot based on the agent type."""

    if agent_type == "neverFoldButPlayRandom":
        neverFoldPolicy = fcpaNeverFoldPolicy.FcpaNeverFoldPolicy(game, [0, 1])
        return PolicyBot(player_id, rng, neverFoldPolicy)
    elif agent_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    elif agent_type == "human":
        return human.HumanBot()
    elif agent_type == "check_call":
        policy = pyspiel.PreferredActionPolicy([1, 0])
        return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
    elif agent_type == "fold":
        policy = pyspiel.PreferredActionPolicy([0, 1])
        return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
    elif agent_type == "50/50_call-fold":
        policy = pyspiel.PreferredActionPolicy([0.5, 0.5])
        return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
    else:
        raise RuntimeError("Unrecognized agent type: {}".format(agent_type))


def generate_states_confidence(game, agents, rng):
    state = game.new_initial_state()

    # Print the initial state
    # print("INITIAL STATE")
    # print(str(state))

    states = [[], []]
    actions_history_int = []

    while not state.is_terminal():
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        current_player = state.current_player()
        if state.is_chance_node():
            # Chance node: sample an outcome
            outcomes = state.chance_outcomes()
            num_actions = len(outcomes)
            # print("Chance node with " + str(num_actions) + " outcomes")
            action_list, prob_list = zip(*outcomes)
            action = rng.choice(action_list, p=prob_list)
            # print("Sampled outcome: ", state.action_to_string(state.current_player(), action))
            state.apply_action(action)

        else:
            states[current_player].append(state.information_state_string())
            # print("state.observation_string()", state.observation_string())
            # Decision node: sample action for the single current player
            legal_actions = state.legal_actions()
            # for action in legal_actions:
            # print("Legal action: {} ({})".format(state.action_to_string(current_player, action), action))
            action = agents[current_player].step(state)
            actions_history_int.append(action)
            # action_string = state.action_to_string(current_player, action)
            # print("Player ", current_player, ", chose action: ",  action_string, "int",action)
            state.apply_action(action)

        # print("")
        # print("NEXT STATE:")
        curr = state.current_player()
        # if (curr != -1 and curr != -4):
        # print("state as string", state.information_state_string())
        # print(str(state))

    # Game is now done. Print utilities for each player
    # print("actions_history_int",actions_history_int)

    returns = state.returns()
    # for pid in range(game.num_players()):
    # print("Utility for player {} is {}".format(pid, returns[pid]))

    # print("playWithoutMoney returned states:", states, "returns", returns)
    return states, returns


def generate_states(game, agents, rng):
    state = game.new_initial_state()

    # Print the initial state
    # print("INITIAL STATE")
    # print(str(state))

    states = []
    actions_history_int = []

    while not state.is_terminal():
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        current_player = state.current_player()
        if state.is_chance_node():
            # Chance node: sample an outcome
            outcomes = state.chance_outcomes()
            num_actions = len(outcomes)
            # print("Chance node with " + str(num_actions) + " outcomes")
            action_list, prob_list = zip(*outcomes)
            action = rng.choice(action_list, p=prob_list)
            # print("Sampled outcome: ", state.action_to_string(state.current_player(), action))
            state.apply_action(action)

        else:
            states.append(state.information_state_string())
            # print("state.information_state_string()", state.information_state_string())
            # Decision node: sample action for the single current player
            legal_actions = state.legal_actions()
            # print("legal_actions",legal_actions, "for player",current_player)
            # for action in legal_actions:
            # print("Legal action: {} ({})".format(state.action_to_string(current_player, action), action))
            action = agents[current_player].step(state)
            actions_history_int.append(action)
            # action_string = state.action_to_string(current_player, action)
            # print("Player ", current_player, "action int",action)
            state.apply_action(action)

        # print("")
        # print("NEXT STATE:")
        curr = state.current_player()
        # if (curr != -1 and curr != -4):
        # print("state as string", state.information_state_string())
        # print(str(state))

    # Game is now done. Print utilities for each player
    # print("actions_history_int",actions_history_int)

    returns = state.returns()
    # for pid in range(game.num_players()):
    # print("Utility for player {} is {}".format(pid, returns[pid]))

    # print("playWithoutMoney returned states:", states, "returns", returns)
    return states, returns, actions_history_int


def create_fcpa_without_money():
    return create_fcpa_with_agents("check_call", "check_call")


def create_fcpa_with_agents(type_agent1: str, type_agent2: str):
    rng = np.random.RandomState(FLAGS.seed)

    games_list = pyspiel.registered_names()
    assert "universal_poker" in games_list

    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    print("Creating game: {}".format(fcpa_game_string))
    game = pyspiel.load_game(fcpa_game_string)

    agents = [
        LoadAgent(type_agent1, game, 0, rng),
        LoadAgent(type_agent2, game, 1, rng)
    ]

    return game, agents, rng


def create_fcpa():
    rng = np.random.RandomState(FLAGS.seed)

    games_list = pyspiel.registered_names()
    assert "universal_poker" in games_list

    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    print("Creating game: {}".format(fcpa_game_string))
    game = pyspiel.load_game(fcpa_game_string)

    return game, rng


def main(_):
    game, rng, agents = create_fcpa_with_agents("random", "random")


if __name__ == "__main__":
    app.run(main)
