# CSE599-O Fall 2025 Assignment 3: Post-Training via RL

For a full description of the assignment, see the assignment handout at
[cse599o_fall2025_assignment3_post_training.pdf](./CSE599_Assignment3.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Acknowledgment

This assignment is adapted from [Assignment 5 of Stanford CS336 (Spring 2025)](https://github.com/stanford-cs336/assignment5-alignment).

