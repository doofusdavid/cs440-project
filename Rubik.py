import copy as copy
import numpy as np
import random

FRONT = 1
LEFT = 0
TOP = 2
BOTTOM = 3
RIGHT = 4
BACK = 5

def printFirstLetters(pair):
    return("{}\t{}\t".format(pair[0][:1],pair[1][:1]))


def printState(state):
    '''
    Prints a Tower of Hanoi state.  Now, with added pegs.
    :param state: list of lists representing tower of hanoi state
    :return: prints out the state nicely.
    '''
    print("\t\t", printFirstLetters(state[TOP][0]))
    print("\t\t", printFirstLetters(state[TOP][1]))
    print(printFirstLetters(state[LEFT][0]), printFirstLetters(state[FRONT][0]),printFirstLetters(state[RIGHT][0]),printFirstLetters(state[BACK][0]))
    print(printFirstLetters(state[LEFT][1]), printFirstLetters(state[FRONT][1]),printFirstLetters(state[RIGHT][1]),printFirstLetters(state[BACK][1]))
    print("\t\t", printFirstLetters(state[BOTTOM][0]))
    print("\t\t", printFirstLetters(state[BOTTOM][1]))
    print("-------------------------------")


def findLongest(state):
    '''
    Finds the longest (highest) peg in Towers of Hanoi.  Used to display in printState
    :param state: list of lists representing tower of hanoi state
    :return: length of longest item
    '''
    length = 0
    for item in state:
        length = max(length, len(item))

    return length


def validMoves(state):
    '''
    Returns a list of lists representing valid 2x2 rubik's cube moves from the given state.
    Move Rotations courtesy: http://www.rubiksplace.com/move-notations/
    :param state: list of lists representing tower of hanoi state
    :return: list of lists representing valid tower of hanoi moves from the given state.
    '''

    # evaluate if we need the front and back moves
    # validStates = ["U", "U'", "D", "D'", "R", "R'", "L", "L'", "F", "F'", "B", "B'"]

    validStates = ["U", "U'", "D", "D'", "R", "R'", "L", "L'"]

    return validStates


def makeMove(state, move, printMoves=False):
    '''
    Takes a move and makes it 2x2 rubik's cube
    :param state: list of lists representing cube
    :param move: possible rubik's cube move
    :return:the state after the move was made
    '''
    newstate = copy.deepcopy(state)
    if move == "U":
        newstate[LEFT][0], newstate[FRONT][0], newstate[RIGHT][0], newstate[BACK][0] = newstate[FRONT][0], newstate[RIGHT][0], newstate[BACK][0], newstate[LEFT][0]
    if move == "U'":
        newstate[FRONT][0], newstate[RIGHT][0], newstate[BACK][0], newstate[LEFT][0] = newstate[LEFT][0], newstate[FRONT][0], newstate[RIGHT][0], newstate[BACK][0]
    if move == "D":
        newstate[LEFT][1], newstate[FRONT][1], newstate[RIGHT][1], newstate[BACK][1] = newstate[FRONT][1], newstate[RIGHT][1], newstate[BACK][1], newstate[LEFT][1]
    if move == "D'":
        newstate[FRONT][1], newstate[RIGHT][1], newstate[BACK][1], newstate[LEFT][1] = newstate[LEFT][1], newstate[FRONT][1], newstate[RIGHT][1], newstate[BACK][1]
    if move == "R":
        newstate[TOP][0][1], newstate[TOP][1][1], newstate[FRONT][0][1], newstate[FRONT][1][1], newstate[BOTTOM][0][1], newstate[BOTTOM][1][1], newstate[BACK][0][1], newstate[BACK][1][1] = newstate[FRONT][0][1], newstate[FRONT][1][1], newstate[BOTTOM][0][1], newstate[BOTTOM][1][1], newstate[BACK][0][1], newstate[BACK][1][1],newstate[TOP][0][1], newstate[TOP][1][1]
    if move == "R'":
        newstate[FRONT][0][1], newstate[FRONT][1][1], newstate[BOTTOM][0][1], newstate[BOTTOM][1][1], newstate[BACK][0][1], newstate[BACK][1][1], newstate[TOP][0][1], newstate[TOP][1][1] = newstate[TOP][0][1], newstate[TOP][1][1], newstate[FRONT][0][1], newstate[FRONT][1][1], newstate[BOTTOM][0][1], newstate[BOTTOM][1][1], newstate[BACK][0][1], newstate[BACK][1][1]
    if move == "L":
        newstate[TOP][0][0], newstate[TOP][1][0], newstate[FRONT][0][0], newstate[FRONT][1][0], newstate[BOTTOM][0][0], newstate[BOTTOM][1][0], newstate[BACK][0][0], newstate[BACK][1][0] = newstate[FRONT][0][0], newstate[FRONT][1][0], newstate[BOTTOM][0][0], newstate[BOTTOM][1][0], newstate[BACK][0][0], newstate[BACK][1][0],newstate[TOP][0][0], newstate[TOP][1][0]
    if move == "L'":
        newstate[FRONT][0][0], newstate[FRONT][1][0], newstate[BOTTOM][0][0], newstate[BOTTOM][1][0], newstate[BACK][0][0], newstate[BACK][1][0], newstate[TOP][0][0], newstate[TOP][1][0] = newstate[TOP][0][0], newstate[TOP][1][0], newstate[FRONT][0][0], newstate[FRONT][1][0], newstate[BOTTOM][0][0], newstate[BOTTOM][1][0], newstate[BACK][0][0], newstate[BACK][1][0]

    if printMoves:
        printState(newstate)

    return newstate


def winner(state):
    '''
    Determines if a winning state occured
    :param state: list of lists representing tower of hanoi state
    :return: True if winning state, False otherwise.
    '''
    completeState = [[["Red", "Red"], ["Red", "Red"]], [["Blue", "Blue"], ["Blue", "Blue"]],
                     [["Yellow", "Yellow"], ["Yellow", "Yellow"]], [["Orange", "Orange"], ["Orange", "Orange"]],
                     [["White", "White"], ["White", "White"]], [["Green", "Green"], ["Green", "Green"]]]

    return (state[LEFT][0]==state[LEFT][1]) and (state[FRONT][0]==state[FRONT][1]) and (state[TOP][0]==state[TOP][1]) and (state[BOTTOM][0]==state[BOTTOM][1]) and (state[RIGHT][0]==state[RIGHT][1]) and (state[BACK][0]==state[BACK][1])
    return state == completeState


def getTuple(state, move):
    '''
    Need immutable type for key to dictionary
    :param state: list of lists representing tower of hanoi state
    :return: tuple representation of the state
    '''
    superTuple = tuple(tuple(tuple(s[0])+tuple(s[1])) for s in state)
    return (superTuple, move)


def epsilonGreedy(epsilon, Q, state, validMovesF):
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
        return tuple(random.choice(goodMoves))
    else:
        # Greedy Move
        Qs = np.array([Q.get(getTuple(state,m), 0.0) for m in goodMoves])
        return tuple(goodMoves[np.argmax(Qs)])


def trainQ(startState, nRepetitions, learningRate, epsilonDecayFactor, validMovesF, makeMoveF):
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
        # reduce the randomness every pass
        epsilon *= epsilonDecayRate
        step = 0
        # hardcoded start state
        state = startState
        done = False

        while not done:
            step += 1
            # grab either a random or best of the known moves
            move = epsilonGreedy(epsilon, Q, state, validMovesF)

            # we don't want to change state directly, and because state is a list of lists, need to do a
            # deepcopy on it, then make the move
            stateNew = copy.deepcopy(state)
            makeMoveF(stateNew, move)

            # if we haven't encountered this state,move combo, add it to Q
            if getTuple(state, move) not in Q:
                Q[getTuple(state, move)] = 0.0  # Initial Q value for new state, move

            # print if debugging
            if showMoves:
                printState(stateNew)
            if winner(stateNew):
                # We won!  backfill Q
                if showMoves:
                    print('End State, we won!')
                Q[getTuple(state, move)] = -1.0
                done = True
                # we're keeping a list of the number of steps it took for each winning solution, so add it here.
                stepList.append(step)

            # update the Q which led us here using the learning factor, and the difference between the current state
            # and the old state
            if step > 1:
                Q[getTuple(stateOld, moveOld)] += rho * (-1 + Q[getTuple(state, move)] - Q[getTuple(stateOld, moveOld)])
                print("Q[{}]: ".format(getTuple(stateOld, moveOld)),Q[getTuple(stateOld, moveOld)])
            # Store the current state, move so we can access it for the next Q update
            stateOld, moveOld = state, move
            state = stateNew

    return Q, stepList


def testQ(Q, maxSteps, validMovesF, makeMoveF):
    '''
    Using the dictionary Q, and the initial state of the game, traverse and return the best path.
    :param Q: dictionary representing the (state,move) - value pairs which, if followed should create the shortest path to the solution.
    :param maxSteps: The number of steps to attempt before giving up.
    :param validMovesF: function returning valid moves of a state
    :param makeMoveF: function making a move on a state
    :return: list containing the states from start to finish
    '''
    state = [[["Red", "Red"],["Red","Red"]],[["Blue", "Blue"],["Blue","Blue"]],[["Yellow", "Yellow"],["Yellow","Yellow"]],[["Orange", "Orange"],["Orange","Orange"]],[["White", "White"],["White","White"]],[["Green", "Green"],["Green","Green"]]]
    statePath = []
    statePath.append(state)

    for i in range(maxSteps):
        if winner(state):
            return statePath
        goodMoves = validMovesF(state)
        Qs = np.array([Q.get(getTuple(state, m), 0.0) for m in goodMoves])
        move = goodMoves[np.argmax(Qs)]
        nextState = copy.deepcopy(state)
        makeMoveF(nextState, move)
        statePath.append(nextState)
        state = nextState

    return "No path found"


completeState = [[["Red", "Red"],["Red","Red"]],[["Blue", "Blue"],["Blue","Blue"]],[["Yellow", "Yellow"],["Yellow","Yellow"]],[["Orange", "Orange"],["Orange","Orange"]],[["White", "White"],["White","White"]],[["Green", "Green"],["Green","Green"]]]

printState(completeState)

newstate = makeMove(completeState, "L")
print(getTuple(newstate,"L'"))

newstate = makeMove(newstate, "L'")


print(getTuple(newstate,"L'"))

newstate = completeState
print(winner(newstate))
for i in range(1):
    newstate = makeMove(newstate,random.choice(validMoves(newstate)))

Q, steps = trainQ(newstate, 10000, 0.5, 0.7, validMoves, makeMove)

print(Q)