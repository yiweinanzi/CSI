from __future__ import annotations

import unittest
from types import SimpleNamespace

from formal_v2.formal_evaluation import (
    _native_response_route_is_active,
    _unified_response_route_is_active,
)


class ResponseEstimandContractTests(unittest.TestCase):
    def test_native_and_unified_response_keep_distinct_route_granularities(self):
        native_key = (0, 0, 1, 2)
        unified_key = (*native_key, 3)
        routed = SimpleNamespace(
            alignment_route={native_key: 2},
            response_route={unified_key: 0},
        )
        self.assertTrue(_native_response_route_is_active(routed, native_key))
        self.assertFalse(_unified_response_route_is_active(routed, unified_key))

        routed = SimpleNamespace(
            alignment_route={native_key: 0},
            response_route={unified_key: 2},
        )
        self.assertFalse(_native_response_route_is_active(routed, native_key))
        self.assertTrue(_unified_response_route_is_active(routed, unified_key))


if __name__ == "__main__":
    unittest.main()
