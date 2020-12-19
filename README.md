![logo](https://github.com/Dobiasd/undictify/raw/master/logo/undictify.png)

[![CI](https://github.com/Dobiasd/undictify/workflows/ci/badge.svg)](https://github.com/Dobiasd/undictify/actions)
[![(License MIT 1.0)](https://img.shields.io/badge/license-MIT%201.0-blue.svg)][license]

[license]: LICENSE


undictify
=========
**Python library providing type-checked function calls at runtime**


Table of contents
-----------------
  * [Introduction](#introduction)
  * [Use case: JSON deserialization](#use-case-json-deserialization)
  * [Details](#details)
  * [Requirements and Installation](#requirements-and-installation)


Introduction
------------
Let's start with a toy example:
```python3
def times_two(value):
    return 2 * value

value = 3
result = times_two(value)
print(f'{value} * 2 == {result}')
```

This is fine, it outputs `output: 3 * 2 = 6`.
But what if `value` accidentally is `'3'` instead of `3`?
The output will become `output: 3 * 2 = 33`, which *might* not be desired.

So you add something like
```python3
if not isinstance(value, int):
    raise TypeError(...)
```
to `times_two`. This will raise an `TypeError` instead, which is better.
But you still only recognize the mistake when actually running the code.
Catching it earlier in the development process might be better.
Luckily Python allows to opt-in for static typing by offering [type annotations](https://docs.python.org/3/library/typing.html).
So you add them and [`mypy`](http://mypy-lang.org/) (or your IDE) will tell you about the problem early.
```python3
def times_two(value: int) -> int:
    return 2 * value

value = '3'
result = times_two(value) # error: Argument 1 to "times_two"
                          # has incompatible type "str"; expected "int"
print(f'{value} * 2 == {result}')
```

But you may get into a situation in which there is no useful static type information,
because of values:
- coming from external non-typed functions (so actually they are of type `Any`)
- were produced by a (rogue) function that returns different types depending on some internal decision (`Union[T, V]`)
- being provided as a `Dict[str, Any]`
- etc.

```python3
def times_two(value: int) -> int:
    return 2 * value
        
def get_value() -> Any:
    return '3'

value = get_value()
result = times_two(value)
print(f'{value} * 2 == {result}')
```

At least with the [appropriate settings](https://stackoverflow.com/questions/51696060/how-to-make-mypy-complain-about-assigning-an-any-to-an-int-part-2/51696314#51696314), `mypy` should dutifully complain, and now you're left with two options:
- Drop type-checking (for example by adding ` # type: ignore` to the end of the `result = times_two(value)` line): This however catapults you back into the insane world where `2 * 3 == 33`.
- You manually add type checks before the call (or inside of `times_two`) like `if not isinstance(value, int):`: This of course does not provide static type checking (because of the dynamic nature of `value`), but at least guarantees sane runtime behavior. 

But the process of writing that boilerplate validation code can become quite cumbersome if you have multiple parameters/functions to check.
Also it is not very [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) since you already have the needed type information in our function signature and you just duplicated it in the check condition.

This is where undictify comes into play. Simply decorate your `times_two` function with `@type_checked_call()`:
```python3
from undictify import type_checked_call

@type_checked_call()
def times_two(value: int) -> int:
    return 2 * value
```

And the arguments of `times_two` will be type-checked with every call at runtime automatically. A `TypeError` will be raised if needed. 

This concept of **runtime type-checks of function calls derived from static type annotations** is quite simple,
however it is very powerful and brings some highly convenient consequences with it.


Use case: JSON deserialization
------------------------------

Imagine your application receives a JSON string representing an entity you need to handle:

```python3
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

tobias = json.loads(tobias_json)
```

Now you start to work with it. Somewhere deep in your business logic you have:
```python3
name_length = len(tobias['name'])
```
But that's only fine if the original JSON string was well-behaved.
If it had `"name": 4,` in it, you would get:
```
    name_length = len(tobias['name'])
TypeError: object of type 'int' has no len()
```
at runtime, which is not nice. So you start to manually add type checking:
```python3
if isinstance(tobias['name'], str):
    name_length = len(tobias['name'])
else:
    # todo: handle the situation somehow
```

You quickly realize that you need to separate concerns better,
in that case the business logic and the input data validation.

So you start to do all checks directly after receiving the data:
```python3
tobias = json.loads(...
if isinstance(tobias['id'], int):
    ...
if isinstance(tobias['name'], str):
    ...
if isinstance(... # *yawn*
```

and then transfer it into a type-safe class instance:
```python3
@dataclass
class Heart:
    weight_in_kg: float
    pulse_at_rest: int

@dataclass
class Human:
    id: int
    name: str
    nick: Optional[str]
    heart: Heart
    friend_ids: List[int]
```

Having the safety provided by the static type annotations (and probably checking your code with `mypy`) is a great because of all the:
- bugs that don't make it into PROD
- manual type checks (and matching unit tests) that you don't have to write
- help your IDE can now offer
- better understanding people get when reading your code
- easier and more confident refactorings

But again, writing all that boilerplate code for data validation is tedious (and not DRY).

So you decide to use a library that does JSON schema validation for you.
But now you have to manually adjust the schema every time your entity structure changes, which still is not DRY, and thus also brings with it all the typical possibilities to make mistakes.

Undictify can help here too!
Annotate the classes `@type_checked_constructor` and their constructors will be wrapped in type-checked calls.
```python3
@type_checked_constructor()
@dataclass
class Heart:
    ...
@type_checked_constructor()
@dataclass
class Human:
    ...
```

(They do not need to be `dataclass`es. Deriving from `NamedTuple` works too.)

Undictify will type-check the construction of objects of type `Heart` and `Human` automatically.
(This works for normal classes with a manually written `__init__` function too.
You just need to provide the type annotations to its parameters.) So you can use the usual dictionary unpacking syntax, to safely convert your untyped dictionary (i.e., `Dict[str, Any]`) resulting from the JSON string into your statically typed class:

```python3
tobias = Human(**json.loads(tobias_json))
```

(Btw this application is the origin of the name of this library.)

It throws exceptions with meaningful details in their associated values in case of errors like:
- missing a field
- a field having the wrong type
- etc.

It also supports optional values being omitted instead of being `None` explicitly (as shown in the example with the `nick` field).


Details
-------

Sometimes, e.g., in case of unpacking a dictionary resulting from a JSON string,
you might want to just skip the fields in the dictionary that your function / constructor does not take as a parameter.
For these cases undictify provides `@type_checked_call(skip=True)`.

It also supports valid type conversions via `@type_checked_call(convert=True)`,
which might for example come in handy when processing the arguments of an HTTP request you receive for example in a `get` handler of a `flask_restful.Resource` class:
```python3
@type_checked_call(convert=True)
def target_function(some_int: int, some_str: str)

class WebController(Resource):
    def get(self) -> Any:
        # request.args is something like {"some_int": "4", "some_str": "hi"}
        result = target_function(**flask.request.args)
```

The values in the `MultiDict` `request.args` are all strings, but the logic behind `@type_checked_call(convert=True)` tries to convert them into the desired target types with reasonable exceptions in case the conversion is not possible.

This way a request to `http://.../foo?some_int=4&some_str=hi` would be handled normally,
but `http://.../foo?some_int=four&some_str=hi` would raise an appropriate `TypeError`.

Additional flexibility is offered for cases in which you would like to not type-check all calls of a specific function / class constructor, but only some. You can use `type_checked_call()` at call site instead of adding the annotation for those:

```python3
from undictify import type_checked_call

def times_two(value: int) -> int:
    return 2 * value

value: Any = '3'
resutl = type_checked_call()(times_two)(value)
```

And last but not least, custom converters for specified parameters are also supported:

```python3
import json
from dataclasses import dataclass
from datetime import datetime

from undictify import type_checked_constructor

def parse_timestamp(datetime_repr: str) -> datetime:
    return datetime.strptime(datetime_repr, '%Y-%m-%dT%H:%M:%SZ')

@type_checked_constructor(converters={'some_timestamp': optional_converter(parse_timestamp)})
@dataclass
class Foo:
    some_timestamp: datetime

json_repr = '{"some_timestamp": "2019-06-28T07:20:34Z"}'
my_foo = Foo(**json.loads(json_repr))
```

In case the converter should be applied even if the source type
already matches the destination type, use `mandatory_converter`
instead of `optional_converter`.


Requirements and Installation
-----------------------------

You need Python 3.7 or higher.

```bash
python3 -m pip install undictify
```

Or, if you like to use latest version from this repository:
```bash
git clone https://github.com/Dobiasd/undictify
cd undictify
python3 -m pip install .
```


License
-------
Distributed under the MIT License.
(See accompanying file [`LICENSE`](https://github.com/Dobiasd/undictify/blob/master/LICENSE) or at
[https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT))
