UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

DANGER_STATES = [
    [False, False, True,  False, False, True,  True,  True],
    [False, False, False, False, False, False, False, True],
    [True,  False, True,  True,  True,  True,  False, True],
    [False, False, True,  True,  True,  True,  False, False],
    [False, False, True,  True,  False, False, False, False],
    [True,  False, True,  True,  False, False, False, True],
    [True,  False, False, False, False, True,  False, False],
    [True,  True,  True,  False, False, True,  False, False],
]
