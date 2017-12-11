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



def validMoves(state):
    '''
    Returns a list of lists representing valid 2x2 rubik's cube moves from the given state.
    Move Rotations courtesy: http://www.rubiksplace.com/move-notations/
    :param state: list of lists representing tower of hanoi state
    :return: list of lists representing valid tower of hanoi moves from the given state.
    '''

    # evaluate if we need the front and back moves
    # validStates = ["U", "U'", "D", "D'", "R", "R'", "L", "L'", "F", "F'", "B", "B'"]

    validStates = ["U", "Uprime", "D", "Dprime", "R", "Rprime", "L", "Lprime"]

    return validStates


def makeMove(state, move, printMoves=False):
    '''
    Takes a move and makes it 2x2 rubik's cube
    :param state: list of lists representing cube
    :param move: possible rubik's cube move
    :return:the state after the move was made
    '''

    if move == "U":
        state[LEFT][0], state[FRONT][0], state[RIGHT][0], state[BACK][0] = state[FRONT][0], state[RIGHT][0], state[BACK][0], state[LEFT][0]
    if move == "Uprime":
        state[FRONT][0], state[RIGHT][0], state[BACK][0], state[LEFT][0] = state[LEFT][0], state[FRONT][0], state[RIGHT][0], state[BACK][0]
    if move == "D":
        state[LEFT][1], state[FRONT][1], state[RIGHT][1], state[BACK][1] = state[FRONT][1], state[RIGHT][1], state[BACK][1], state[LEFT][1]
    if move == "Dprime":
        state[FRONT][1], state[RIGHT][1], state[BACK][1], state[LEFT][1] = state[LEFT][1], state[FRONT][1], state[RIGHT][1], state[BACK][1]
    if move == "R":
        state[TOP][0][1], state[TOP][1][1], state[FRONT][0][1], state[FRONT][1][1], state[BOTTOM][0][1], state[BOTTOM][1][1], state[BACK][0][1], state[BACK][1][1] = state[FRONT][0][1], state[FRONT][1][1], state[BOTTOM][0][1], state[BOTTOM][1][1], state[BACK][0][1], state[BACK][1][1],state[TOP][0][1], state[TOP][1][1]
    if move == "Rprime":
        state[FRONT][0][1], state[FRONT][1][1], state[BOTTOM][0][1], state[BOTTOM][1][1], state[BACK][0][1], state[BACK][1][1], state[TOP][0][1], state[TOP][1][1] = state[TOP][0][1], state[TOP][1][1], state[FRONT][0][1], state[FRONT][1][1], state[BOTTOM][0][1], state[BOTTOM][1][1], state[BACK][0][1], state[BACK][1][1]
    if move == "L":
        state[TOP][0][0], state[TOP][1][0], state[FRONT][0][0], state[FRONT][1][0], state[BOTTOM][0][0], state[BOTTOM][1][0], state[BACK][0][0], state[BACK][1][0] = state[FRONT][0][0], state[FRONT][1][0], state[BOTTOM][0][0], state[BOTTOM][1][0], state[BACK][0][0], state[BACK][1][0],state[TOP][0][0], state[TOP][1][0]
    if move == "Lprime":
        state[FRONT][0][0], state[FRONT][1][0], state[BOTTOM][0][0], state[BOTTOM][1][0], state[BACK][0][0], state[BACK][1][0], state[TOP][0][0], state[TOP][1][0] = state[TOP][0][0], state[TOP][1][0], state[FRONT][0][0], state[FRONT][1][0], state[BOTTOM][0][0], state[BOTTOM][1][0], state[BACK][0][0], state[BACK][1][0]

    if printMoves:
        printState(state)

    return state


def winner(state):
    '''
    Determines if a winning state occured
    :param state: list of lists representing tower of hanoi state
    :return: True if winning state, False otherwise.
    '''
    completeState = [[["Red", "Red"], ["Red", "Red"]], [["Blue", "Blue"], ["Blue", "Blue"]],
                     [["Yellow", "Yellow"], ["Yellow", "Yellow"]], [["Orange", "Orange"], ["Orange", "Orange"]],
                     [["White", "White"], ["White", "White"]], [["Green", "Green"], ["Green", "Green"]]]

    return faceComplete(state[LEFT]) and faceComplete(state[FRONT]) and faceComplete(state[TOP]) and faceComplete(state[BOTTOM]) and faceComplete(state[RIGHT]) and faceComplete(state[BACK])

    # return state == completeState

def faceComplete(face):
    return (face[0][0]== face[0][1] == face[1][0] == face[1][1])


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
        return random.choice(goodMoves)
    else:
        # Greedy Move
        Qs = np.array([Q.get(getTuple(state,m), 0.0) for m in goodMoves])
        return goodMoves[np.argmax(Qs)]


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
        # if nGames % 10 == 0: print(".", end="")
        # if nGames % 100 == 0: print("Q length: ", len(Q))
        # reduce the randomness every pass
        epsilon *= epsilonDecayRate
        step = 0
        # hardcoded start state
        state = startState
        done = False

        while not done:
            #if step % 100 == 0: print(".", end="")
            #if step % 1000 == 0: print("Q length: ", len(Q))
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
                # print('End State, we won!')
                Q[getTuple(state, move)] = -1.0
                done = True
                # we're keeping a list of the number of steps it took for each winning solution, so add it here.
                stepList.append(step)

            # update the Q which led us here using the learning factor, and the difference between the current state
            # and the old state
            if step > 1:
                Q[getTuple(stateOld, moveOld)] += rho * (-1 + Q[getTuple(state, move)] - Q[getTuple(stateOld, moveOld)])
                #print("Q[",getTuple(stateOld, moveOld),"]: ",Q[getTuple(stateOld, moveOld)])
            # Store the current state, move so we can access it for the next Q update
            stateOld, moveOld = state, move
            state = stateNew

    return Q, stepList


def testQ(Q, initialState, maxSteps, validMovesF, makeMoveF):
    '''
    Using the dictionary Q, and the initial state of the game, traverse and return the best path.
    :param Q: dictionary representing the (state,move) - value pairs which, if followed should create the shortest path to the solution.
    :param maxSteps: The number of steps to attempt before giving up.
    :param validMovesF: function returning valid moves of a state
    :param makeMoveF: function making a move on a state
    :return: list containing the states from start to finish
    '''
    #state = [[["Red", "Red"],["Red","Red"]],[["Blue", "Blue"],["Blue","Blue"]],[["Yellow", "Yellow"],["Yellow","Yellow"]],[["Orange", "Orange"],["Orange","Orange"]],[["White", "White"],["White","White"]],[["Green", "Green"],["Green","Green"]]]
    state = initialState

    statePath = []
    movePath = []
    movePath.append("Initial")
    statePath.append(state)

    for i in range(maxSteps):
        if winner(state):
            return statePath, movePath
        goodMoves = validMovesF(state)
        Qs = np.array([Q.get(getTuple(state, m), -1000.0) for m in goodMoves])
        move = goodMoves[np.argmax(Qs)]
        movePath.append(move)
        nextState = copy.deepcopy(state)
        makeMoveF(nextState, move)
        statePath.append(nextState)
        state = nextState

    return "No path found",movePath

#
completeState = [[["Red", "Red"],["Red","Red"]],[["Blue", "Blue"],["Blue","Blue"]],[["Yellow", "Yellow"],["Yellow","Yellow"]],[["Orange", "Orange"],["Orange","Orange"]],[["White", "White"],["White","White"]],[["Green", "Green"],["Green","Green"]]]
newstate = completeState
for i in range(30):
    move = random.choice(validMoves(newstate))
    print("Move ", i, " was: ", move)
    newstate = makeMove(newstate,move)


printState(newstate)
startTime = time.time()
Q, steps = trainQ(newstate, 2000, 0.5, 0.7, validMoves, makeMove)
endTime = time.time()
print(steps)
path, moveList = testQ(Q, newstate, 20000, validMoves, makeMove)

print("Training took: ", endTime-startTime, " seconds.")
print("Mean of solution length: ", np.mean(steps))
print("Median of solution length: ", np.median(steps))
print("Q length:", len(Q))
if path == "No path found":
    print(path)
else:
    for i in range(len(path)):
        printState(path[i])
        print("Move: ", moveList[i])
