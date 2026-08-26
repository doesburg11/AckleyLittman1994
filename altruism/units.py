"""Unit layout for the 32-unit linear-threshold network (Figure 1).

Units 0-12 are INPUT units: their values are set directly from the
environment (or fixed, for TRUE) every step, never computed from a
weighted sum. Units 13-31 are COMPUTED units: linear-threshold, output
0/1, activated when the weighted sum of their incoming connections is
> 0 -- these are the ones the two-pass synchronous update actually
touches.

Exact index assignment per Figure 1's unit table:
  0: True (always 1)
  1: Pred, 2: Food                     -- stimulus sensors
  3: @L, 4: @1, 5: @2, 6: @R            -- location sensors (one-hot)
  7-12: >a..>f                          -- hearing sensors (0..8 each)
  13-18: <A..<F                         -- speech effectors
  19: Move, 20: To R                    -- movement effectors
  21-31: hidden (11 units, no prespecified function)
"""

TRUE = 0
PRED = 1
FOOD = 2
AT_L = 3
AT_1 = 4
AT_2 = 5
AT_R = 6
HEAR = list(range(7, 13))     # 7..12, channels a..f
SPEAK = list(range(13, 19))   # 13..18, channels A..F
MOVE = 19
TO_R = 20
HIDDEN = list(range(21, 32))  # 21..31

N_UNITS = 32
N_INPUT_UNITS = 13   # 0..12
N_COMPUTED_UNITS = 19  # 13..31 (8 effector + 11 hidden)

LOCATIONS = ["L", "1", "2", "R"]
LOCATION_UNIT = {"L": AT_L, "1": AT_1, "2": AT_2, "R": AT_R}


def is_input_unit(unit: int) -> bool:
    """True/Pred/Food/location/hearing units -- set from environment,
    never computed. Also the guard used by the genome's destination-mode
    mode-shift code (weight_specifier == 7): growing an incoming
    connection into one of these is meaningless, since their value is
    always overwritten by the environment regardless of any accumulated
    weighted input."""
    return unit < N_INPUT_UNITS
