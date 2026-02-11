"""Backward-compatibility redirect — imports all tests from test_intentional_bridge.

This ensures the old test file name still works with any CI that references it.
"""

from tests.graph.test_intentional_bridge import *  # noqa: F401, F403
