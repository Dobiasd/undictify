#!/usr/bin/env python3

"""
undictify - examples from README.md
"""

import json
from typing import List, NamedTuple, Optional

from undictify import unpack_dict

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2018, Tobias Hermann"
__email__ = "editgym@gmail.com"
__license__ = "MIT"


class Heart(NamedTuple):
    weight_in_kg: float
    pulse_at_rest: int


class Human(NamedTuple):
    id: int
    name: str
    nick: Optional[str]
    heart: Heart
    friends: List[int]


def main():
    json_data_tobi = '''
        {
            "id": 1,
            "name": "Tobias",
            "nick": "Tobi",
            "heart": {
                "weight_in_kg": 0.31,
                "pulse_at_rest": 52
            },
            "friends": [2, 3, 4, 5]
        }'''

    json_data_katrin = '''
        {
            "id": 2,
            "name": "Katrin",
            "heart": {
                "weight_in_kg": 0.28,
                "pulse_at_rest": 57
            },
            "friends": [1, 3, 6, 7, 8]
        }'''

    unsafe_tobi = Human(**json.loads(json_data_tobi))
    tobi: Human = unpack_dict(Human, json.loads(json_data_tobi))
    katrin: Human = unpack_dict(Human, json.loads(json_data_katrin))
    assert unsafe_tobi.name == 'Tobias'
    assert tobi.heart.pulse_at_rest == 52
    assert katrin.nick is None
    assert len(katrin.friends) == 5


if __name__ == "__main__":
    main()
