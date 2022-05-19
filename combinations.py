def get_combinations_of_cards(listOfCards):
    cardNumbers = listOfCards[0:len(listOfCards):2]
    cardColors = listOfCards[1:len(listOfCards):2]
    twoOfNumber = 0.0
    threeOfAKind = 0.0
    fourOfAKind = 0.0
    straight = 0.0
    color = 0.0

    for card in cardNumbers:
        if cardNumbers.count(card) == 2:
            twoOfNumber += 0.5 + card * 0.1 / 2
            if cardNumbers[0] == card or cardNumbers[1] == card:
                twoOfNumber = twoOfNumber * 20
        if cardNumbers.count(card) == 3:
            threeOfAKind += 1.0 / 3 + card * 0.1 / 3
            if cardNumbers[0] == card or cardNumbers[1] == card:
                threeOfAKind += 100.0 / 3
        if cardNumbers.count(card) == 4:
            fourOfAKind += 1.0 / 4 + card * 0.1 / 4
            if cardNumbers[0] == card or cardNumbers[1] == card:
                fourOfAKind += 100.0 / 4
        straight = get_straight(cardNumbers)

    for card in cardColors:
        if cardColors.count(card) >= 5:
            color += 1.0 / 5
            if cardColors[0] == card or cardColors[1] == card:
                color += 100.0 / 5

    listOfCards.append(twoOfNumber)
    listOfCards.append(threeOfAKind)
    listOfCards.append(fourOfAKind)
    listOfCards.append(straight)
    listOfCards.append(color)
    listOfCards.append(float(cardNumbers[0]))
    # return listOfCards
    return [twoOfNumber, threeOfAKind, fourOfAKind, straight, color, float(cardNumbers[0])]


def get_straight(cardNumbers):
    seq = 1
    higher = 15.0
    ret = 0
    prev = -2.0
    output = sorted(cardNumbers, key=lambda x: float(x))
    for card in output:
        if card == 0.0 and float(12) in cardNumbers:
            seq = 2
        else:
            if card == prev + 1.0:
                seq += 1
                higher = card
            else:
                seq = 1
        if seq == 5:
            if higher - 5 < cardNumbers[0] <= higher:
                ret += 100.0
            if higher - 5 < cardNumbers[1] <= higher:
                ret += 100.0
            ret += 1.0
            return ret
        prev = card
    return 0.0


def main():
    print(get_combinations_of_cards([12.0, 2.0, 11.0, 2.0, 10.0, 2.0, 8.0, 2.0, 9.0, 2.0, 3.0, 3.0, 5.0, 9.0]))


if __name__ == "__main__":
    main()
