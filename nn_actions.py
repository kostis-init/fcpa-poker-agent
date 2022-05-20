from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
import numpy as np
from open_spiel.python.bots.policy import PolicyBot
import nn
import parse_state
from nn_confidence_level import applyNN_confidence_level, loadNN_confidence_level
import fcpaPolicy
import play_fcpa
import keras


def create_data_set(nbGames, game, agents, rng, models_confidence):
    # 6 inputs: round_i, confidence_level, last_action, pot, remaining_money_me, remaining_money_other
    max_entries = nbGames * 2 * 4 * 10
    data_set_in = np.zeros((max_entries, 6), dtype=float)  # nb data_in nbGames * 2 players * 4 rounds
    # 4 outputs: 4 possible actions = ['f','c','p','a'] -> [1,2,3,4]
    data_set_out = np.zeros((max_entries, 4), dtype=float)

    print("data_set_in.shape:", data_set_in.shape)
    print("data_set_out.shape:", data_set_out.shape)
    print("nbGames:", nbGames)

    index_to_fill = 0
    for i in range(nbGames):
        states, returns, actions_history = play_fcpa.generate_states(game, agents, rng)
        index_to_fill = state_to_training_data_fills(
            states, returns, actions_history, data_set_in, data_set_out, models_confidence, index_to_fill)

    data_set_in = data_set_in[:index_to_fill]
    data_set_out = data_set_out[:index_to_fill]
    print("data_set_in:", data_set_in.shape)
    print("data_set_out:", data_set_out.shape)

    return data_set_in, data_set_out


def utilities_to_floats(utilities):
    if utilities[0] > utilities[1]:
        output = 1.0
    elif utilities[0] == utilities[1]:
        output = 0.5
    else:
        output = 0.0

    return output, 1 - output


def get_winner_id(utilities: [int]) -> int:
    if utilities[0] > utilities[1]:
        winner_i = 0
    elif utilities[0] < utilities[1]:
        winner_i = 1
    else:  # draw
        winner_i = -1
    return winner_i


# TODO check if change this
def utilities_to_floats_actions(utilities: [int], player_i: int, max_reward: int, action_performed_int: int) -> [float]:
    action_floats = [0.5 for i in range(4)]
    reward = float(utilities[player_i])
    winner_i = get_winner_id(utilities)

    if action_performed_int == 0:  # fold
        action_floats = [0.0 for i in range(4)]
        return action_floats

    if winner_i == player_i:  # WIN
        if action_performed_int == 1:  # check
            action_floats = [0.0, 1.0, 0.65, 0.5]
        if action_performed_int == 2:  # pot raise
            action_floats = [0.0, 0.8, 1.0, 0.5]
        if action_performed_int == 1:  # all-in
            action_floats = [0.0, 0.5, 0.8, 1.0]
    else:  # LOSE
        action_floats = [1.0, 0.0, 0.0, 0.0]

    return action_floats


def get_input_from_state(round_i, state, models_confidence_level) -> [int]:
    # 6 inputs: round_i, confidence_level, last_action, pot, remaining_money_me, remaining_money_other
    input_list = [round_i]

    encoded_cards = parse_state.get_cards_from_state(state)
    confidence_level = applyNN_confidence_level(models_confidence_level, encoded_cards)
    input_list.append(confidence_level)

    input_list.append(parse_state.get_last_action_from_state(state))
    input_list.append(parse_state.get_pot_from_state(state))
    wallets = parse_state.get_wallets_from_state(state)
    input_list.append(wallets[0])
    input_list.append(wallets[1])

    return input_list


def state_to_training_data_fills(flat_states, utilities, actions_history: [int], data_set_in, data_set_out,
                                 models_confidence, index_to_fill):
    # print("flat_states", flat_states)
    # print("actions_history", actions_history)

    state_index = 0
    current_player = 0
    while state_index < len(flat_states):
        # INIT
        current_state = flat_states[state_index]
        round_index = parse_state.get_round_from_state(current_state)
        current_player = parse_state.get_last_player_id(current_state)
        action_to_perform = actions_history[state_index]
        # print("roundIndex:", round_index, "current_player", current_player, "action_to_perform", action_to_perform, current_state)

        MAXREWARD = sum(parse_state.get_wallets_from_state(current_state))  # TODO to modify to something else

        # GET INPUT
        # 6 inputs: round_i, confidence_level, last_action, pot, remaining_money_me, remaining_money_other
        state_info = get_input_from_state(round_index, current_state, models_confidence)
        # print("dataSet_in line player1:", state_info, " from state", current_state)

        # GET OUTPUT
        # 4 outputs: 4 possible actions = ['f','c','p','a'] -> [1,2,3,4]
        float_reward = utilities_to_floats_actions(utilities, current_player, MAXREWARD, action_to_perform)
        # print("dataSet_out", float_reward, "with action_performed", action_to_perform, "and utilities", utilities)

        # INSERT DATASET
        # put the data in the matrices data_set
        data_set_in[index_to_fill] = state_info
        data_set_out[index_to_fill] = float_reward

        index_to_fill += 1
        state_index += 1

    return index_to_fill


def train_nn_never_folds(nb_games_training, models_confidence):
    game, agents, rng = play_fcpa.create_fcpa_with_agents("neverFoldButPlayRandom", "neverFoldButPlayRandom")

    data_sets_in, data_set_out = create_data_set(nb_games_training, game, agents, rng, models_confidence)

    return nn.get_model_agent(data_sets_in, data_set_out)


def train_nn_randoms(nb_games_training, models_confidence):
    game, agents, rng = play_fcpa.create_fcpa_with_agents("random", "random")

    data_sets_in, data_set_out = create_data_set(nb_games_training, game, agents, rng, models_confidence)

    return nn.get_model_agent(data_sets_in, data_set_out)


def get_agent_from_nn(nn_model, models_confidence_level, player_id):
    game, rng = play_fcpa.create_fcpa()
    fcpa_policy = fcpaPolicy.FcpaPolicy(game, [0, 1], nn_model, models_confidence_level)
    policy_bot = PolicyBot(player_id, rng, fcpa_policy)
    return policy_bot


def train_nn_adversarial(nb_games_training, agent1, agent2, models_confidence):
    game, rng = play_fcpa.create_fcpa()

    data_sets_in, data_set_out = create_data_set(nb_games_training, game, [agent1, agent2], rng, models_confidence)

    return nn.get_model_agent(data_sets_in, data_set_out)


def apply_nn(nn_model, state, modules_confidence):
    round_index = parse_state.get_round_from_state(state)
    input_state = get_input_from_state(round_index, state, modules_confidence)

    np_input = np.array([input_state])
    actions = nn_model.predict(np_input)[0]

    return actions


def load_nn():
    # return keras.models.load_model('./models/model_actions')
    package_directory = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(package_directory, 'models', 'model_actions.h5')
    print("Model folder: " + str(model_folder))
    return keras.models.load_model(model_folder)


def save_nn(nn_model):
    nn_model.save('./models/model_actions')


def test_nn_on_state(nn_model, state, models_confidence_level):
    return apply_nn(nn_model, state, models_confidence_level)


def run_all_tests(nn_model, models_confidence_level):
    state = '[Round 2][Player: 0][Pot: 200][Money: 19900 19900][Private: 9s4d][Public: Jd7s5c3h][Sequences: cc|cc|]'
    result = test_nn_on_state(nn_model, state, models_confidence_level)
    print("returned:", result, "for state", state)


def evaluate_against_random_agents(agent1, nb_games):
    game, agents_random, rng = play_fcpa.create_fcpa_with_agents("random", "random")
    random_agent = agents_random[1]

    total_utilities = [0, 0]
    for i_game in range(nb_games):
        states, utilities, history_actions = play_fcpa.generate_states(game, [agent1, random_agent], rng)
        total_utilities[0] += utilities[0]
        total_utilities[1] += utilities[1]

    return [total_utilities[0] / nb_games, total_utilities[1] / nb_games]


def main(_):
    # Make sure poker is compiled into the library, as it requires an optional
    # dependency: the ACPC poker code. To ensure it is compiled in, prepend both
    # the install.sh and build commands with OPEN_SPIEL_BUILD_WITH_ACPC=ON.
    # See here:
    # https://github.com/deepmind/open_spiel/blob/master/docs/install.md#configuration-conditional-dependencies
    # for more details on optional dependencies.

    models_confidence = loadNN_confidence_level()

    print("Loaded models confidence level")

    model_from_randoms_1 = train_nn_never_folds(100, models_confidence)
    model_from_randoms_2 = train_nn_never_folds(100, models_confidence)

    print("TRAINED agents against random")

    run_all_tests(model_from_randoms_1, models_confidence)

    policy_agent1 = get_agent_from_nn(model_from_randoms_1, models_confidence, 0)
    policy_agent2 = get_agent_from_nn(model_from_randoms_2, models_confidence, 1)

    print("AGENTS created")
    print("TRAIN ADVERSARIAL")

    best_model = train_nn_adversarial(100, policy_agent1, policy_agent2, models_confidence)
    best_policy_agent = get_agent_from_nn(best_model, models_confidence, 0)

    print("TRAINED best_model")
    print("EVALUATE")

    results_utilities = evaluate_against_random_agents(best_policy_agent, 100)
    print("Evaluate results_utilities", results_utilities)

    return


if __name__ == "__main__":
    app.run(main)
