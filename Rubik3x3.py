import copy as copy
import numpy as np
import random
import time

FRONT = 1
LEFT = 0
TOP = 2
BOTTOM = 3
RIGHT = 4
BACK = 5


def printFirstLetters3x3(pair):
    return ("{}\t{}\t{}\t".format(pair[0][:1], pair[1][:1], pair[2][:1]))


def printState3x3(state):
    '''
    Prints a Tower of Hanoi state.  Now, with added pegs.
    :param state: list of lists representing tower of hanoi state
    :return: prints out the state nicely.
    '''
    print("\t\t\t", printFirstLetters3x3(state[TOP][0]))
    print("\t\t\t", printFirstLetters3x3(state[TOP][1]))
    print("\t\t\t", printFirstLetters3x3(state[TOP][2]))
    print(printFirstLetters3x3(state[LEFT][0]), printFirstLetters3x3(state[FRONT][0]),
          printFirstLetters3x3(state[RIGHT][0]), printFirstLetters3x3(state[BACK][0]))
    print(printFirstLetters3x3(state[LEFT][1]), printFirstLetters3x3(state[FRONT][1]),
          printFirstLetters3x3(state[RIGHT][1]), printFirstLetters3x3(state[BACK][1]))
    print(printFirstLetters3x3(state[LEFT][2]), printFirstLetters3x3(state[FRONT][2]),
          printFirstLetters3x3(state[RIGHT][2]), printFirstLetters3x3(state[BACK][2]))
    print("\t\t\t", printFirstLetters3x3(state[BOTTOM][0]))
    print("\t\t\t", printFirstLetters3x3(state[BOTTOM][1]))
    print("\t\t\t", printFirstLetters3x3(state[BOTTOM][2]))
    print("-------------------------------")


def makeMove3x3(state, move, printMoves=False):
    '''
    Takes a move and makes it 2x2 rubik's cube
    :param state: list of lists representing cube
    :param move: possible rubik's cube move
    :return:the state after the move was made
    '''

    if move == "U":
        state[LEFT][0], state[FRONT][0], state[RIGHT][0], state[BACK][0] = state[FRONT][0], state[RIGHT][0], \
                                                                           state[BACK][0], state[LEFT][0]
    if move == "Uprime":
        state[FRONT][0], state[RIGHT][0], state[BACK][0], state[LEFT][0] = state[LEFT][0], state[FRONT][0], \
                                                                           state[RIGHT][0], state[BACK][0]
    if move == "D":
        state[LEFT][2], state[FRONT][2], state[RIGHT][2], state[BACK][2] = state[FRONT][2], state[RIGHT][2], \
                                                                           state[BACK][2], state[LEFT][2]
    if move == "Dprime":
        state[FRONT][2], state[RIGHT][2], state[BACK][2], state[LEFT][2] = state[LEFT][2], state[FRONT][2], \
                                                                           state[RIGHT][2], state[BACK][2]
    if move == "R":
        state[TOP][0][2], state[TOP][1][2], state[TOP][2][2], state[FRONT][0][2], state[FRONT][1][2], state[FRONT][2][
            2], state[BOTTOM][0][2], state[BOTTOM][1][2], state[BOTTOM][2][2], state[BACK][0][2], state[BACK][1][2], \
        state[BACK][2][2] = state[FRONT][0][2], state[FRONT][1][2], state[FRONT][2][2], state[BOTTOM][0][2], \
                            state[BOTTOM][1][2], state[BOTTOM][2][2], state[BACK][0][2], state[BACK][1][2], \
                            state[BACK][2][2], state[TOP][0][2], state[TOP][1][2], state[TOP][2][2]
    if move == "Rprime":
        state[FRONT][0][2], state[FRONT][1][2], state[FRONT][2][2], state[BOTTOM][0][2], state[BOTTOM][1][2], \
        state[BOTTOM][2][2], state[BACK][0][2], state[BACK][1][2], state[BACK][2][2], state[TOP][0][2], state[TOP][1][
            2], state[TOP][2][2] = state[TOP][0][2], state[TOP][1][2], state[TOP][2][2], state[FRONT][0][2], \
                                   state[FRONT][1][2], state[FRONT][2][2], state[BOTTOM][0][2], state[BOTTOM][1][2], \
                                   state[BOTTOM][2][2], state[BACK][0][2], state[BACK][1][2], state[BACK][2][2]
    if move == "L":
        state[TOP][0][0], state[TOP][1][0], state[TOP][2][0], state[FRONT][0][0], state[FRONT][1][0], state[FRONT][2][
            0], state[BOTTOM][0][0], state[BOTTOM][1][0], state[BOTTOM][2][0], state[BACK][0][0], state[BACK][1][0], \
        state[BACK][2][0] = state[FRONT][0][0], state[FRONT][1][0], state[FRONT][2][0], state[BOTTOM][0][0], \
                            state[BOTTOM][1][0], state[BOTTOM][2][0], state[BACK][0][0], state[BACK][1][0], \
                            state[BACK][2][0], state[TOP][0][0], state[TOP][1][0], state[TOP][2][0]
    if move == "Lprime":
        state[FRONT][0][0], state[FRONT][1][0], state[FRONT][2][0], state[BOTTOM][0][0], state[BOTTOM][1][0], \
        state[BOTTOM][2][0], state[BACK][0][0], state[BACK][1][0], state[BACK][2][0], state[TOP][0][0], state[TOP][1][
            0], state[TOP][2][0] = state[TOP][0][0], state[TOP][1][0], state[TOP][2][0], state[FRONT][0][0], \
                                   state[FRONT][1][0], state[FRONT][2][0], state[BOTTOM][0][0], state[BOTTOM][1][0], \
                                   state[BOTTOM][2][0], state[BACK][0][0], state[BACK][1][0], state[BACK][2][0]

    if printMoves:
        printState3x3(state)

    return state


def winner3x3(state):
    '''
    Determines if a winning state occured
    :param state: list of lists representing tower of hanoi state
    :return: True if winning state, False otherwise.
    '''

    return faceComplete3x3(state[LEFT]) and faceComplete3x3(state[FRONT]) and faceComplete3x3(
        state[TOP]) and faceComplete3x3(
        state[BOTTOM]) and faceComplete3x3(state[RIGHT]) and faceComplete3x3(state[BACK])


def faceComplete3x3(face):
    return (
    face[0][0] == face[0][1] == face[0][2] == face[1][0] == face[1][1] == face[1][2] == face[2][0] == face[2][1] ==
    face[2][2])


def epsilonGreedy(epsilon, Q, state, validMovesF, tupleF):
    '''
    Makes either a random move, or tries the move which Q indicates is the best.
    :param epsilon: A decreasing number representing the level of randomness
    :param Q: Dictionary of state,move - value pairs, with the higher values being better moves
    :param state: list of lists representing tower of hanoi state
    :param validMovesF: function returning valid moves
    :return:
    '''
    goodMoves = validMovesF(state)
    if np.random.uniform() < epsilon:
        # Random Move
        return random.choice(goodMoves)
    else:
        # Greedy Move
        Qs = np.array([Q.get(tupleF(state, m), 0.0) for m in goodMoves])
        return goodMoves[np.argmax(Qs)]


def trainQ(startState, nRepetitions, learningRate, epsilonDecayFactor, validMovesF, makeMoveF, winnerF, tupleF):
    '''
    Creates and fills a dictionary, Q, representing the (state,move) - value pairs which, if followed
    should create the shortest path to the solution.
    :param nRepetitions: how many times to iterate through.  Higher numbers would generate more accurate results
    :param learningRate: How much to adjust the value part of the dictionary
    :param epsilonDecayFactor: how quickly to reduce the random factor.
    :param validMovesF: function returning valid moves of a state
    :param makeMoveF: function making a move on a state
    :return: the dictionary, Q, and a list containing the number of steps it took per iteration to find the goal state
    '''
    maxGames = nRepetitions
    rho = learningRate
    epsilonDecayRate = epsilonDecayFactor
    epsilon = 1.0
    Q = {}
    stepList = []
    # show the moves while debuggin
    showMoves = False

    for nGames in range(maxGames):
        # if nGames % 10 == 0: print(".", end="")
        # if nGames % 100 == 0: print("Q length: ", len(Q))
        # reduce the randomness every pass
        epsilon *= epsilonDecayRate
        step = 0
        # hardcoded start state
        state = startState
        done = False

        while not done:
            # if step % 100 == 0: print(".", end="")
            # if step % 1000 == 0: print("Q length: ", len(Q))
            step += 1
            # grab either a random or best of the known moves
            move = epsilonGreedy(epsilon, Q, state, validMovesF, tupleF)

            # we don't want to change state directly, and because state is a list of lists, need to do a
            # deepcopy on it, then make the move
            stateNew = copy.deepcopy(state)
            makeMoveF(stateNew, move)

            # if we haven't encountered this state,move combo, add it to Q
            if tupleF(state, move) not in Q:
                Q[tupleF(state, move)] = 0.0  # Initial Q value for new state, move

            # print if debugging
            if showMoves:
                printState(stateNew)
            if winnerF(stateNew):
                # We won!  backfill Q
                # print('End State, we won!')
                Q[tupleF(state, move)] = -1.0
                done = True
                # we're keeping a list of the number of steps it took for each winning solution, so add it here.
                stepList.append(step)

            # update the Q which led us here using the learning factor, and the difference between the current state
            # and the old state
            if step > 1:
                Q[tupleF(stateOld, moveOld)] += rho * (-1 + Q[tupleF(state, move)] - Q[tupleF(stateOld, moveOld)])
            # Store the current state, move so we can access it for the next Q update
            stateOld, moveOld = state, move
            state = stateNew

    return Q, stepList


def testQ(Q, initialState, maxSteps, validMovesF, makeMoveF, winnerF, tupleF):
    '''
    Using the dictionary Q, and the initial state of the game, traverse and return the best path.
    :param Q: dictionary representing the (state,move) - value pairs which, if followed should create the shortest path to the solution.
    :param maxSteps: The number of steps to attempt before giving up.
    :param validMovesF: function returning valid moves of a state
    :param makeMoveF: function making a move on a state
    :return: list containing the states from start to finish
    '''
    state = initialState

    statePath = []
    movePath = []
    movePath.append("Initial")
    statePath.append(state)

    for i in range(maxSteps):
        if winnerF(state):
            return statePath, movePath
        goodMoves = validMovesF(state)
        Qs = np.array([Q.get(tupleF(state, m), -1000.0) for m in goodMoves])
        move = goodMoves[np.argmax(Qs)]
        movePath.append(move)
        nextState = copy.deepcopy(state)
        makeMoveF(nextState, move)
        statePath.append(nextState)
        state = nextState

    return "No path found", movePath


def validMoves(state):
    '''
    Returns a list of lists representing valid rubik's cube moves from the given state.
    Move Rotations courtesy: http://www.rubiksplace.com/move-notations/
    :param state: list of lists representing tower of hanoi state
    :return: list of lists representing valid tower of hanoi moves from the given state.
    '''

    validStates = ["U", "Uprime", "D", "Dprime", "R", "Rprime", "L", "Lprime"]

    return validStates


def getTuple3x3(state, move):
    '''
    Need immutable type for key to dictionary
    :param state: list of lists representing tower of hanoi state
    :return: tuple representation of the state
    '''
    superTuple = tuple(tuple(tuple(s[0]) + tuple(s[1]) + tuple(s[2])) for s in state)
    return (superTuple, move)


completeState = [[["Red", "Red", "Red"],["Red","Red", "Red"],["Red","Red", "Red"]],[["Blue", "Blue","Blue"],["Blue", "Blue","Blue"],["Blue", "Blue","Blue"]],[["Yellow", "Yellow", "Yellow"],["Yellow", "Yellow", "Yellow"],["Yellow", "Yellow", "Yellow"]],[["Orange", "Orange", "Orange"],["Orange", "Orange", "Orange"],["Orange", "Orange", "Orange"]],[["White", "White", "White"],["White", "White", "White"],["White", "White", "White"]],[["Green", "Green", "Green"],["Green", "Green", "Green"],["Green", "Green", "Green"]]]
newstate = completeState
for i in range(1):
    move = random.choice(validMoves(newstate))
    print("Move ", i, " was: ", move)
    newstate = makeMove3x3(newstate,move)


printState3x3(newstate)

startTime = time.time()
Q, steps = trainQ(newstate, 10, 0.5, 0.7, validMoves, makeMove3x3, winner3x3, getTuple3x3)
endTime = time.time()
print(steps)
path, moveList = testQ(Q, newstate, 20000, validMoves, makeMove3x3, winner3x3, getTuple3x3)

print("Training took: ", endTime-startTime, " seconds.")
print("Mean of solution length: ", np.mean(steps))
print("Median of solution length: ", np.median(steps))
print("Q length:", len(Q))
if path == "No path found":
    print(path)
else:
    for i in range(len(path)):
        printState3x3(path[i])
        print("Move: ", moveList[i])

