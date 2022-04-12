UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

DANGER_STATES = [
    [False, False, False,  False, False, True,  True,  True],
    [False, False, False, False, False, False, False, True],
    [False,  False, True,  False,  False,  True,  False, True],
    [False, False, False,  True,  False,  False,  False, False],
    [False, False, False,  True,  True, False, False, False],
    [False,  False, True,  False,  False, True, False, False],
    [False,  True, False, False, False, False,  True, False],
    [True,  False,  False,  False, False, False,  False, False],
]

TERMINAL = 'terminal'
SAFE = 'safe'
DANGER = 'danger'
