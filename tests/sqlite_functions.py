# -*- coding: utf-8 -*-
"""Tests for per-DAL SQLite aggregate and window function registration."""

import sqlite3
import tempfile
import threading

from pydal import DAL
from pydal.exceptions import NotOnSQLiteError

from ._adapt import IS_SQLITE
from ._compat import unittest


class SlidingWeightedAverage:
    def __init__(self):
        self.weighted_sum = 0.0
        self.total_weight = 0.0

    def step(self, value, weight):
        self.weighted_sum += value * weight
        self.total_weight += weight

    def inverse(self, value, weight):
        self.weighted_sum -= value * weight
        self.total_weight -= weight

    def value(self):
        if not self.total_weight:
            return None
        return self.weighted_sum / self.total_weight

    def finalize(self):
        return self.value()


class DistinctCount:
    def __init__(self):
        self.seen = set()

    def step(self, value):
        self.seen.add(value)

    def finalize(self):
        return len(self.seen)


class DistinctCountTwoArgs:
    def __init__(self):
        self.seen = set()

    def step(self, first, second):
        self.seen.add((first, second))

    def finalize(self):
        return len(self.seen)


class ReplacedDistinctCount:
    def __init__(self):
        self.count = 0

    def step(self, value):
        self.count += 1

    def finalize(self):
        return -self.count


class FinalizeFailure:
    def step(self, value):
        pass

    def finalize(self):
        raise RuntimeError("finalize failed")


@unittest.skipUnless(IS_SQLITE, "SQLite custom functions only")
class TestSQLiteFunctions(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.uri = "sqlite://custom-functions.db"

    def make_db(self):
        db = DAL(self.uri, folder=self.tempdir.name, migrate=False)
        self.addCleanup(db.close)
        return db

    def create_data(self, db):
        db.executesql(
            "CREATE TABLE measurements ("
            "grp TEXT, seq INTEGER, value REAL, weight REAL)"
        )
        db._adapter.connection.executemany(
            "INSERT INTO measurements VALUES (?, ?, ?, ?)",
            [
                ("a", 1, 10.0, 1),
                ("a", 2, 20.0, 3),
                ("a", 3, 30.0, 1),
                ("b", 1, 8.0, 2),
            ],
        )
        db.commit()

    def test_aggregate_and_window_functions_survive_reconnection(self):
        db = self.make_db()
        db.register_sqlite_window_function(
            "sliding_weighted_average", 2, SlidingWeightedAverage
        )
        db.register_sqlite_aggregate("distinct_count", 1, DistinctCount)

        self.create_data(db)
        distinct = db.executesql(
            "SELECT grp, distinct_count(seq) FROM measurements "
            "GROUP BY grp ORDER BY grp"
        )
        self.assertEqual(distinct, [("a", 3), ("b", 1)])

        windowed = db.executesql(
            "SELECT grp, seq, sliding_weighted_average(value, weight) OVER ("
            "PARTITION BY grp ORDER BY seq ROWS BETWEEN 1 PRECEDING AND CURRENT ROW"
            ") FROM measurements ORDER BY grp, seq"
        )
        self.assertEqual(
            windowed,
            [
                ("a", 1, 10.0),
                ("a", 2, 17.5),
                ("a", 3, 22.5),
                ("b", 1, 8.0),
            ],
        )

        first_connection = db._adapter.connection
        db._adapter.reconnect()
        self.assertIsNot(db._adapter.connection, first_connection)
        distinct = db.executesql(
            "SELECT grp, distinct_count(seq) FROM measurements "
            "GROUP BY grp ORDER BY grp"
        )
        self.assertEqual(distinct, [("a", 3), ("b", 1)])
        windowed = db.executesql(
            "SELECT grp, seq, sliding_weighted_average(value, weight) OVER ("
            "PARTITION BY grp ORDER BY seq ROWS BETWEEN 1 PRECEDING AND CURRENT ROW"
            ") FROM measurements ORDER BY grp, seq"
        )
        self.assertEqual(windowed[1], ("a", 2, 17.5))

    def test_registration_is_available_on_two_alternating_connections(self):
        db = self.make_db()
        db.register_sqlite_window_function(
            "sliding_weighted_average", 2, SlidingWeightedAverage
        )
        self.create_data(db)

        first_connection_id = id(db._adapter.connection)
        first_result = db.executesql(
            "SELECT sliding_weighted_average(value, weight) OVER ("
            "PARTITION BY grp ORDER BY seq ROWS BETWEEN 1 PRECEDING AND CURRENT ROW"
            ") FROM measurements WHERE grp = 'a' ORDER BY seq"
        )
        self.assertEqual(first_result[0], (10.0,))

        ready = threading.Event()
        finished = threading.Event()
        worker_state = {}

        def use_second_connection():
            ready.wait()
            try:
                worker_state["connection_id"] = id(db._adapter.connection)
                worker_state["result"] = db.executesql(
                    "SELECT sliding_weighted_average(value, weight) OVER ("
                    "PARTITION BY grp ORDER BY seq "
                    "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) "
                    "FROM measurements WHERE grp = 'b' ORDER BY seq"
                )
            except Exception as exc:
                worker_state["error"] = exc
            finally:
                db._adapter.close()
                finished.set()

        worker = threading.Thread(target=use_second_connection)
        worker.start()
        ready.set()
        finished.wait(5)
        worker.join(5)

        self.assertIsNone(worker_state.get("error"))
        self.assertNotEqual(
            worker_state["connection_id"], first_connection_id
        )
        self.assertEqual(worker_state["result"], [(8.0,),])

        final_result = db.executesql(
            "SELECT sliding_weighted_average(value, weight) OVER ("
            "PARTITION BY grp ORDER BY seq ROWS BETWEEN 1 PRECEDING AND CURRENT ROW"
            ") FROM measurements WHERE grp = 'a' ORDER BY seq"
        )
        self.assertEqual(id(db._adapter.connection), first_connection_id)
        self.assertEqual(final_result[2], (22.5,))

    def test_same_name_different_argument_counts_are_independent(self):
        db = self.make_db()
        db.register_sqlite_aggregate("distinct_count", 1, DistinctCount)
        db.register_sqlite_aggregate("distinct_count", 2, DistinctCountTwoArgs)
        self.create_data(db)

        result = db.executesql(
            "SELECT distinct_count(seq), distinct_count(grp, seq) FROM measurements"
        )
        self.assertEqual(result, [(3, 4)])

    def test_callable_factory_gets_independent_state_for_each_aggregate(self):
        def distinct_count_factory():
            seen = set()

            class CallbackDistinctCount:
                def step(self, value):
                    seen.add(value)

                def finalize(self):
                    return len(seen)

            return CallbackDistinctCount()

        db = self.make_db()
        db.register_sqlite_aggregate(
            "callback_distinct_count", 1, distinct_count_factory
        )
        self.create_data(db)

        self.assertEqual(
            db.executesql(
                "SELECT grp, callback_distinct_count(seq) FROM measurements "
                "GROUP BY grp ORDER BY grp"
            ),
            [("a", 3), ("b", 1)],
        )

    def test_replacement_and_unregistration_apply_to_later_statements(self):
        db = self.make_db()
        db.register_sqlite_aggregate("distinct_count", 1, DistinctCount)
        self.create_data(db)
        self.assertEqual(
            db.executesql("SELECT distinct_count(seq) FROM measurements"), [(3,)]
        )

        db.register_sqlite_aggregate(
            "distinct_count", 1, ReplacedDistinctCount, replace=True
        )
        self.assertEqual(
            db.executesql("SELECT distinct_count(seq) FROM measurements"), [(-4,)]
        )

        db.unregister_sqlite_aggregate("distinct_count", 1)
        with self.assertRaises(sqlite3.OperationalError):
            db.executesql("SELECT distinct_count(seq) FROM measurements")
        db._adapter.reconnect()
        with self.assertRaises(sqlite3.OperationalError):
            db.executesql("SELECT distinct_count(seq) FROM measurements")
        self.assertEqual(db.executesql("SELECT 1"), [(1,)])

    def test_finalize_failure_reports_database_exception_and_connection_recovers(
        self,
    ):
        db = self.make_db()
        db.register_sqlite_aggregate("finalize_failure", 1, FinalizeFailure)
        self.create_data(db)

        with self.assertRaises(sqlite3.OperationalError) as caught:
            db.executesql("SELECT finalize_failure(seq) FROM measurements")
        self.assertIn("finalize", str(caught.exception).lower())
        self.assertEqual(db.executesql("SELECT 1"), [(1,)])

    def test_other_dal_instances_are_not_affected(self):
        registered = self.make_db()
        unregistered = self.make_db()
        registered.register_sqlite_aggregate("distinct_count", 1, DistinctCount)
        self.create_data(registered)
        registered.commit()

        self.assertEqual(
            registered.executesql(
                "SELECT distinct_count(seq) FROM measurements"
            ),
            [(3,)],
        )
        with self.assertRaises(sqlite3.OperationalError):
            unregistered.executesql("SELECT distinct_count(seq) FROM measurements")

    def test_window_function_registration_failure_is_raised_at_registration(self):
        db = self.make_db()
        adapter = db._adapter
        original_version_info = adapter.driver.sqlite_version_info
        adapter.driver.sqlite_version_info = (3, 24, 0)
        try:
            with self.assertRaises(
                (sqlite3.NotSupportedError, NotImplementedError)
            ):
                db.register_sqlite_window_function(
                    "sliding_weighted_average", 2, SlidingWeightedAverage
                )
        finally:
            adapter.driver.sqlite_version_info = original_version_info

        self.assertNotIn(
            ("sliding_weighted_average", 2, "window"),
            adapter._sqlite_functions,
        )


class TestSQLiteFunctionsBackendCheck(unittest.TestCase):
    def test_sqlite_function_registration_is_rejected_for_non_sqlite_dal(self):
        db = DAL(None)
        with self.assertRaises(NotOnSQLiteError):
            db.register_sqlite_aggregate("distinct_count", 1, DistinctCount)
        with self.assertRaises(NotOnSQLiteError):
            db.unregister_sqlite_aggregate("distinct_count", 1)


if __name__ == "__main__":
    unittest.main()
