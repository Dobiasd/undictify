"""
undictify - tests with PEP 563
"""
from __future__ import annotations  # pylint: disable=no-name-in-module,misplaced-future

import json
import unittest
from typing import NamedTuple

from ._unpack import type_checked_constructor


@type_checked_constructor(convert=False, skip=False)  # pylint: disable=too-few-public-methods
class WithOneMember(NamedTuple):
    """Some dummy class as a NamedTuple."""
    val: int


@type_checked_constructor(convert=True, skip=True)  # pylint: disable=too-few-public-methods
class WithOneMemberSkipConv(NamedTuple):
    """Some dummy class as a NamedTuple."""
    val: int


class TestArgsCalls(unittest.TestCase):
    """Tests function calls with positional and keywords arguments."""

    def test_simple(self) -> None:
        """Just one check to see if it works nonetheless"""
        result = WithOneMember(val=42)
        self.assertEqual(result.val, 42)

    def test_skip_conv(self) -> None:
        """Just one check to see if it works nonetheless"""
        object_repr = '{"val": "42", "to_skip": "skip"}'
        result = WithOneMemberSkipConv(**json.loads(object_repr))
        self.assertEqual(result.val, 42)
