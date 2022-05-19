from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras.models
from absl import app
import numpy as np
import parse_state
from format_mappings import encode_hand
import nn
import play_fcpa


def createDataSetConfidence(nbGames, game, agents, rng):
    # list_dataSets_in = [np.zeros((nbGames * 2, 2*i+6), dtype=float) for i in [2, 5, 6, 7]]
    list_dataSets_in = [np.zeros((nbGames * 2, 6), dtype=float) for i in [2, 5, 6, 7]]
    dataSet_out = np.zeros(nbGames * 2, dtype=float)

    for dataSet in list_dataSets_in:
        print("dataSetsIn.shape:", dataSet.shape)
    print("dataSetsIn.shape:", dataSet_out.shape)

    for i in range(nbGames):
        states, returns = play_fcpa.generate_states_confidence(game, agents, rng)
        stateToTrainingDataFills(states, returns, list_dataSets_in, dataSet_out, i)
        # states, returns = playWithoutMoney(game, agents, rng)

    return list_dataSets_in, dataSet_out


def stateToTrainingDataFills(states, utilities, list_dataSet_in, dataSet_out, gameIndex):
    # print("STATES")
    player0States = states[0]
    player1States = states[1]

    for roundIndex in range(4):
        # print("round i:", roundIndex)
        # print("public", publicCards, "private cards", privateCards0, privateCards1)

        if utilities[0] > utilities[1]:
            output = 1.0
        elif utilities[0] == utilities[1]:
            output = 0.5
        else:
            output = 0.0
        # CHECK THAT PRIVATE IS BEFORE PUBLIC
        cards1 = parse_state.get_encoded_cards_from_state(player0States[roundIndex])
        cards2 = parse_state.get_encoded_cards_from_state(player1States[roundIndex])
        # cards2 = encode_hand(np.concatenate([privateCards1, publicCards]))

        # print("cards1:",cards1)

        list_dataSet_in[roundIndex][2 * gameIndex] = cards1
        list_dataSet_in[roundIndex][(2 * gameIndex) + 1] = cards2

        dataSet_out[2 * gameIndex] = output
        dataSet_out[(2 * gameIndex) + 1] = 1 - output

    # print("utilities:", utilities)

    return


# def createFcpaWithoutMoney():
#     rng = np.random.RandomState(FLAGS.seed)
#
#     games_list = pyspiel.registered_names()
#     assert "universal_poker" in games_list
#
#     fcpa_game_string = pyspiel.hunl_game_string("fcpa")
#     print("Creating game: {}".format(fcpa_game_string))
#     game = pyspiel.load_game(fcpa_game_string)
#
#     agents = [
#         LoadAgent("check_call", game, 0, rng),
#         LoadAgent("check_call", game, 1, rng)
#     ]
#
#     return game, agents, rng


def trainNN_confidence_level(nb_games_training):
    game, agents, rng = play_fcpa.create_fcpa_without_money()

    list_dataSets_in, dataSet_out = createDataSetConfidence(nb_games_training, game, agents, rng)

    y = dataSet_out

    models = []
    for round_i in range(4):
        X = list_dataSets_in[round_i]
        # print("NN x shape:", X.shape)
        # print("NN input shape:", X[0].shape)

        # print("NN y shape:", y.shape)
        # save model goes here
        model = nn.get_model_confidence_level(X, y)
        # model.save(/...)
        models.append(model)

    return models


"""
train just one of the NN confidence level
"""


def trainNN_round_confidence_level(round_i, nb_games_training):
    game, agents, rng = play_fcpa.create_fcpa_without_money()

    list_dataSets_in, dataSet_out = createDataSetConfidence(nb_games_training, game, agents, rng)

    y = dataSet_out

    X = list_dataSets_in[round_i]

    model = nn.get_model_confidence_level(X, y)
    return model


def evaluateNN_round_confidence_level(round_i, model, nb_games_test):
    game, agents, rng = play_fcpa.create_fcpa_without_money()

    list_test_in, test_out = createDataSetConfidence(nb_games_test, game, agents, rng)

    # evaluate the keras model
    print("EVALUATE")

    results = model.evaluate(list_test_in[round_i], test_out, verbose=1)

    _, accuracy = results
    print("Accuracy:", str(accuracy))

    return results


def evaluateNN_confidence_level(models, nb_games_test):
    game, agents, rng = play_fcpa.create_fcpa_without_money()

    list_test_in, test_out = createDataSetConfidence(nb_games_test, game, agents, rng)

    results = []
    for round_i in range(4):
        model = models[round_i]

        # evaluate the keras model
        print("EVALUATE")
        results.append(model.evaluate(list_test_in[round_i], test_out, verbose=1))

    return results


def applyNN_confidence_level(models, cards: [str]) -> float:
    cards_per_round = np.array([2, 5, 6, 7])
    nb_cards = len(cards)
    round_i = np.where(nb_cards == cards_per_round)[0][0]
    encoded_cards = encode_hand(cards)

    # print("applyNN_confidence_level",cards_per_round, nb_cards, round_i, encoded_cards, models)

    model = models[round_i]

    confidence_level = model.predict([encoded_cards])

    return confidence_level[0][0]


def loadNN_confidence_level():
    models = []
    package_directory = os.path.dirname(os.path.abspath(__file__))
    for i in range(4):
        model_folder = os.path.join(package_directory, 'models', 'model_' + str(i) + '.h5')
        # models.append(keras.models.load_model('./models/model_' + str(i)))
        models.append(keras.models.load_model(model_folder))
    return models


def test_models_on_cards(models, cards):
    confidence_level_cards = applyNN_confidence_level(models, cards)
    print("apply NN_confidence_level to goodcards:", cards, " NN returns:", confidence_level_cards)
    return


def run_all_tests(models):
    test_models_on_cards(models, ['Ah', 'As', 'Ad', '5s', '5h'])
    test_models_on_cards(models, ['5h', '3s', '8d', '5s', '5h'])
    test_models_on_cards(models, ['Ah', 'Ks', 'Jd', 'Qs', 'Th'])
    test_models_on_cards(models, ['Qh', '4s', '6d', '9s', '2h'])
    test_models_on_cards(models, ['2h', '4s', 'Ad', 'As', 'Ah'])
    test_models_on_cards(models, ['Th', '7h', '2h', '3h', '5h'])
    test_models_on_cards(models, ['Ah', 'Ad'])
    test_models_on_cards(models, ['2h', '7d'])
    test_models_on_cards(models, ['2h', 'As', 'Th', '3h', '2h', '5s', '5s'])
    test_models_on_cards(models, ['2h', 'As', '2h', '3h', '2h', '5s', '5s'])
    test_models_on_cards(models, ['2h', 'As', 'Th', '3h', '2h', '5s'])
    test_models_on_cards(models, ['2h', 'As', 'Th', '3h', '2h'])
    test_models_on_cards(models, ['2h', '3h', '8d', '5h', 'As', '7s', '4s'])
    return


def main(_):
    # Make sure poker is compiled into the library, as it requires an optional
    # dependency: the ACPC poker code. To ensure it is compiled in, prepend both
    # the install.sh and build commands with OPEN_SPIEL_BUILD_WITH_ACPC=ON.
    # See here:
    # https://github.com/deepmind/open_spiel/blob/master/docs/install.md#configuration-conditional-dependencies
    # for more details on optional dependencies.

    # model_7 = trainNN_round_confidence_level(3, 1000)
    # result = evaluateNN_round_confidence_level(3, model_7, 100)

    # models = trainNN_confidence_level(20000)

    loaded_models = trainNN_confidence_level(20000)
    run_all_tests(loaded_models)

    # print(result)
    # game, agents, rng = play_fcpa.createFcpaWithoutMoney()

    # statesBothPlayers = play_fcpa.generate_states(game, agents, rng)
    # print("statesBothPlayers",statesBothPlayers)

    return


if __name__ == "__main__":
    app.run(main)
