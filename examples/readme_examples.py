#!/usr/bin/env python3

"""
undictify - examples from README.md
"""

import json
from datetime import datetime
from typing import List, NamedTuple, Optional, Any

from undictify import type_checked_constructor, optional_converter

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


@type_checked_constructor(skip=True)
class Heart(NamedTuple):
    weight_in_kg: float
    pulse_at_rest: int


@type_checked_constructor(skip=True)
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


def parse_timestamp(datetime_repr: str) -> datetime:
    return datetime.strptime(datetime_repr, '%Y-%m-%dT%H:%M:%SZ')


@type_checked_constructor(converters={'some_timestamp': optional_converter(parse_timestamp)})
class Foo(NamedTuple):
    some_timestamp: datetime


def custom_converter_1():
    json_repr = '{"some_timestamp": "2019-06-28T07:20:34Z"}'
    my_foo = Foo(**json.loads(json_repr))
    print(my_foo)


def main():
    intro_1()
    intro_2()
    intro_3()
    json_1()
    custom_converter_1()


if __name__ == "__main__":
    main()
