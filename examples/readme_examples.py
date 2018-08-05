#!/usr/bin/env python3

"""
undictify - examples from README.md
"""

import json
from typing import List, NamedTuple, Optional, Any, TypeVar, Callable

from undictify import type_checked_apply, type_checked_apply_convert
from undictify import type_checked_call

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2018, Tobias Hermann"
__email__ = "editgym@gmail.com"
__license__ = "MIT"


def intro_1():
    def times_two(value):
        return 2 * value

    value = 3
    result = times_two(value)
    print(f'{value} * 2 == {result}')


def intro_2():
    def times_two(value):
        return 2 * value

    value = '3'
    result = times_two(value)
    print(f'{value} * 2 == {result}')


def intro_3():
    def times_two(value: int) -> int:
        return 2 * value

    def get_value() -> Any:
        return '3'

    value = get_value()
    result = times_two(value)
    print(f'{value} * 2 == {result}')


@type_checked_call
class Heart(NamedTuple):
    weight_in_kg: float
    pulse_at_rest: int


@type_checked_call
class Human(NamedTuple):
    id: int
    name: str
    nick: Optional[str]
    heart: Heart
    friend_ids: List[int]


def json_1():
    tobias_json = '''
        {
            "id": 1,
            "name": "Tobias",
            "heart": {
                "weight_in_kg": 0.31,
                "pulse_at_rest": 52
            },
            "friend_ids": [2, 3, 4, 5]
        }'''
    tobias = Human(**json.loads(tobias_json))
    assert len(tobias.friend_ids) == 4


TypeT = TypeVar('TypeT')


def unpack_json(target_func: Callable[..., TypeT],
                object_repr: str, convert_types: bool = False) -> TypeT:
    if convert_types:
        return type_checked_apply_convert(target_func,
                                          **json.loads(object_repr))
    return type_checked_apply(target_func,
                              **json.loads(object_repr))


def json_2():
    tobias_json = '''
        {
            "id": 1,
            "name": "Tobias",
            "heart": {
                "weight_in_kg": 0.31,
                "pulse_at_rest": 52
            },
            "friend_ids": [2, 3, 4, 5]
        }'''
    tobias = unpack_json(Human, tobias_json)
    assert len(tobias.friend_ids) == 4


def main():
    intro_1()
    intro_2()
    intro_3()
    json_1()
    json_2()


if __name__ == "__main__":
    main()
