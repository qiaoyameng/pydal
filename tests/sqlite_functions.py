# -*- coding: utf-8 -*-
"""Tests for DAL-managed SQLite aggregate and window functions."""

import os
import sqlite3
import tempfile

from pydal import DAL

from ._adapt import IS_SQLITE
from ._compat import unittest


def weighted_average_step(state, value, weight):
    total_weight, weighted_total = state
    return total_weight + weight, weighted_total + value * weight


def weighted_average_inverse(state, value, weight):
    total_weight, weighted_total = state
    return total_weight - weight, weighted_total - value * weight


def weighted_average_value(state):
    total_weight, weighted_total = state
    return weighted_total / total_weight


def weighted_average_initial():
    return 0.0, 0.0


def distinct_step(values, value):
    values.add(value)


def make_window_class(result):
    class TestWindow:
        def __init__(self):
            self.seen = 0

        def step(self, value):
            self.seen += 1

        def inverse(self, value):
            self.seen -= 1

        def value(self):
            return result

        def finalize(self):
            return None

    return TestWindow


@unittest.skipUnless(IS_SQLITE, "SQLite-specific tests")
class TestSQLiteFunctions(unittest.TestCase):
    def setUp(self):
        self.databases = []
        self.temp_paths = []

    def tearDown(self):
        for db in self.databases:
            db.close()
        for path in self.temp_paths:
            for suffix in ("", "-journal", "-wal", "-shm"):
                try:
                    os.unlink(path + suffix)
                except OSError:
                    pass

    def connect(self, uri="sqlite:memory"):
        db = DAL(uri)
        self.databases.append(db)
        return db

    def temp_file_uri(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            path = temp.name
        self.temp_paths.append(path)
        return "sqlite://%s" % path, path

    def create_measurements(self, db):
        db.executesql("CREATE TABLE measurements (g INTEGER, x INTEGER, w INTEGER)")
        rows = [(1, 1, 1), (1, 2, 2), (2, 3, 1), (2, 3, 3), (2, 4, 1)]
        for row in rows:
            db.executesql("INSERT INTO measurements VALUES (?, ?, ?)", row)
        db.commit()

    def register_weighted_average(self, db, replace=False):
        db.register_sqlite_window_function(
            "weighted_average",
            2,
            step=weighted_average_step,
            inverse=weighted_average_inverse,
            value=weighted_average_value,
            initial=weighted_average_initial,
            replace=replace,
        )

    def register_distinct_count(self, db, name="distinct_count", replace=False):
        db.register_sqlite_aggregate(
            name,
            1,
            step=distinct_step,
            finalize=len,
            initial=set,
            replace=replace,
        )

    def test_window_and_group_aggregate_queries(self):
        db = self.connect()
        self.create_measurements(db)
        self.register_weighted_average(db)
        self.register_distinct_count(db)

        averages = db.executesql(
            """
            SELECT g, x,
                   weighted_average(x, w) OVER (
                       PARTITION BY g ORDER BY x
                       ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
                   )
            FROM measurements
            ORDER BY g, x, w
            """
        )
        self.assertEqual(averages[0][:2], (1, 1))
        self.assertAlmostEqual(averages[0][2], 1.0)
        self.assertAlmostEqual(averages[1][2], 5.0 / 3.0)
        self.assertEqual(averages[2][:2], (2, 3))
        self.assertAlmostEqual(averages[2][2], 3.0)
        self.assertAlmostEqual(averages[3][2], 3.0)
        self.assertAlmostEqual(averages[4][2], 3.25)

        counts = db.executesql(
            "SELECT g, distinct_count(x) FROM measurements GROUP BY g ORDER BY g"
        )
        self.assertEqual(counts, [(1, 2), (2, 2)])

    def test_aggregate_class_has_independent_state(self):
        class CountAggregate:
            def __init__(self):
                self.count = 0

            def step(self, value):
                self.count += 1

            def finalize(self):
                return self.count

        db = self.connect()
        db.executesql("CREATE TABLE values_table (g INTEGER, x INTEGER)")
        for row in ((1, 1), (1, 1), (1, 2), (2, 7)):
            db.executesql("INSERT INTO values_table VALUES (?, ?)", row)
        db.commit()
        db.register_sqlite_aggregate("class_count", 1, CountAggregate)

        self.assertEqual(
            db.executesql(
                "SELECT g, class_count(x) FROM values_table GROUP BY g ORDER BY g"
            ),
            [(1, 3), (2, 1)],
        )
        self.assertEqual(
            db.executesql("SELECT class_count(x) FROM values_table"), [(4,)]
        )

    def test_registration_survives_reconnection(self):
        uri, path = self.temp_file_uri()
        db = self.connect(uri)
        self.register_distinct_count(db)
        db.executesql("CREATE TABLE values_table (g INTEGER, x INTEGER)")
        db.executesql("INSERT INTO values_table VALUES (1, 1)")
        db.executesql("INSERT INTO values_table VALUES (1, 2)")
        db.commit()

        self.assertEqual(
            db.executesql("SELECT distinct_count(x) FROM values_table"), [(2,)]
        )
        db._adapter.reconnect()
        self.assertEqual(
            db.executesql("SELECT distinct_count(x) FROM values_table"), [(2,)]
        )
        db.close()
        self.assertEqual(
            db.executesql("SELECT distinct_count(x) FROM values_table"), [(2,)]
        )

    def test_two_databases_alternate_without_sharing_registrations(self):
        uri, path = self.temp_file_uri()
        first = self.connect(uri)
        second = self.connect(uri)
        first.executesql("CREATE TABLE values_table (x INTEGER)")
        first.executesql("INSERT INTO values_table VALUES (1)")
        first.executesql("INSERT INTO values_table VALUES (1)")
        first.executesql("INSERT INTO values_table VALUES (2)")
        first.commit()

        self.register_distinct_count(first)
        self.assertEqual(
            first.executesql("SELECT distinct_count(x) FROM values_table"), [(2,)]
        )
        with self.assertRaises(sqlite3.OperationalError):
            second.executesql("SELECT distinct_count(x) FROM values_table")

        second.register_sqlite_aggregate(
            "distinct_count",
            1,
            step=lambda values, value: values + 1,
            finalize=lambda values: values * 10,
            initial=int,
        )
        self.assertEqual(
            second.executesql("SELECT distinct_count(x) FROM values_table"), [(30,)]
        )
        self.assertEqual(
            first.executesql("SELECT distinct_count(x) FROM values_table"), [(2,)]
        )
        self.assertEqual(
            second.executesql("SELECT distinct_count(x) FROM values_table"), [(30,)]
        )

    def test_same_name_with_different_argument_counts(self):
        db = self.connect()
        self.create_measurements(db)
        db.register_sqlite_aggregate(
            "overload",
            1,
            step=lambda state, x: state + x,
            finalize=lambda state: state,
            initial=int,
        )
        db.register_sqlite_aggregate(
            "overload",
            2,
            step=lambda state, x, y: state + x * y,
            finalize=lambda state: state,
            initial=int,
        )

        self.assertEqual(db.executesql("SELECT overload(x) FROM measurements"), [(13,)])
        self.assertEqual(db.executesql("SELECT overload(x, w) FROM measurements"), [(21,)])

        with self.assertRaises(ValueError):
            db.register_sqlite_aggregate(
                "overload", 1, step=lambda state, x: state, initial=int
            )
        db.unregister_sqlite_aggregate("overload", 1)
        with self.assertRaises(sqlite3.OperationalError):
            db.executesql("SELECT overload(x) FROM measurements")
        self.assertEqual(db.executesql("SELECT overload(x, w) FROM measurements"), [(21,)])

    def test_replaces_existing_aggregate_with_same_signature(self):
        db = self.connect()
        db.executesql("CREATE TABLE values_table (x INTEGER)")
        for value in (1, 2):
            db.executesql("INSERT INTO values_table VALUES (?)", (value,))
        db.commit()
        db.register_sqlite_aggregate(
            "same_signature",
            1,
            step=lambda state, value: state + value,
            finalize=lambda state: state,
            initial=int,
        )
        self.assertEqual(db.executesql("SELECT same_signature(x) FROM values_table"), [(3,)])

        db.register_sqlite_aggregate(
            "same_signature",
            1,
            step=lambda state, value: state + value * 10,
            finalize=lambda state: state,
            initial=int,
            replace=True,
        )
        self.assertEqual(db.executesql("SELECT same_signature(x) FROM values_table"), [(30,)])

    def test_finalize_failure_uses_database_error_path_and_connection_recovers(self):
        db = self.connect()
        db.executesql("CREATE TABLE values_table (x INTEGER)")
        db.executesql("INSERT INTO values_table VALUES (1)")
        db.commit()

        def failing_finalize(state):
            raise RuntimeError("finalize failed")

        db.register_sqlite_aggregate(
            "failing_finalize",
            1,
            step=lambda state, value: state + 1,
            finalize=failing_finalize,
            initial=int,
        )
        with self.assertRaises(sqlite3.OperationalError) as caught:
            db.executesql("SELECT failing_finalize(x) FROM values_table")
        self.assertIn("finalize", str(caught.exception))
        self.assertEqual(db.executesql("SELECT 1"), [(1,)])
        with self.assertRaises(sqlite3.OperationalError):
            db.executesql("SELECT failing_finalize(x) FROM values_table")

    def test_replacement_and_unregistration_apply_after_current_statement(self):
        db = self.connect()
        db.executesql("CREATE TABLE values_table (x INTEGER)")
        for value in (1, 2, 3):
            db.executesql("INSERT INTO values_table VALUES (?)", (value,))
        db.commit()
        db.register_sqlite_window_function(
            "versioned", 1, window_class=make_window_class(1)
        )

        cursor = db._adapter.connection.execute(
            "SELECT versioned(x) OVER () FROM values_table"
        )
        self.assertEqual(cursor.fetchone(), (1,))
        db.register_sqlite_window_function(
            "versioned", 1, window_class=make_window_class(2), replace=True
        )
        self.assertEqual(cursor.fetchall(), [(1,), (1,)])
        self.assertEqual(
            db.executesql("SELECT versioned(x) OVER () FROM values_table LIMIT 1"),
            [(2,)],
        )

        cursor = db._adapter.connection.execute(
            "SELECT versioned(x) OVER () FROM values_table"
        )
        self.assertEqual(cursor.fetchone(), (2,))
        db.unregister_sqlite_window_function("versioned", 1)
        self.assertEqual(cursor.fetchall(), [(2,), (2,)])
        with self.assertRaises(sqlite3.OperationalError):
            db.executesql("SELECT versioned(x) OVER () FROM values_table")

    def test_window_support_is_checked_during_registration(self):
        db = self.connect()

        class UnsupportedConnection:
            def __init__(self, *args, **kwargs):
                pass

            def close(self):
                pass

        original_connection = db._adapter.driver.Connection
        db._adapter.driver.Connection = UnsupportedConnection
        try:
            with self.assertRaises(NotImplementedError):
                db.register_sqlite_window_function(
                    "unsupported", 1, window_class=make_window_class(1)
                )
        finally:
            db._adapter.driver.Connection = original_connection
        self.assertNotIn(
            ("window", "unsupported", 1), db._adapter._sqlite_functions
        )


if __name__ == "__main__":
    unittest.main()
