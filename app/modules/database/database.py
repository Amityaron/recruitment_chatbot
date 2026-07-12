"""
Database Module
---------------
Creates and manages a SQL Server database (schema mirrors db_Tech.sql).

Table: dbo.Schedule
    ScheduleID  - unique row ID (IDENTITY)
    date        - interview date (DATE)
    time        - interview time (TIME)
    position    - job position (e.g. 'Python Dev')
    available   - 1 = free slot, 0 = taken (BIT)

Connection: Windows Authentication against SQL_SERVER (default: localhost).
Run create_and_seed_db() once to create and populate the database.
At runtime, use get_available_slots() and book_slot().
"""

import os
import random
import pyodbc
from datetime import date, timedelta, time as dt_time
from dotenv import load_dotenv

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
SQL_SERVER   = os.getenv("SQL_SERVER", "localhost")
SQL_DATABASE = os.getenv("SQL_DATABASE", "Tech")

# ── Constants (mirror db_Tech.sql) ───────────────────────────────────────────
POSITIONS   = ["Python Dev", "Sql Dev", "Analyst", "ML"]
WORK_HOURS  = ["09:00", "10:00", "11:00", "12:00", "13:00",
               "14:00", "15:00", "16:00", "17:00"]
# Sun=6, Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5
# db_Tech.sql: Tue-Fri & Sun only (skip Saturday and Monday)
VALID_WEEKDAYS = {6, 1, 2, 3, 4}   # Sun, Tue, Wed, Thu, Fri


# ── Connection ────────────────────────────────────────────────────────────────

def _connection_string(database: str) -> str:
    return (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
    )


def get_connection():
    """Return a connection to the Tech SQL Server database."""
    return pyodbc.connect(_connection_string(SQL_DATABASE))


# ── Database creation ─────────────────────────────────────────────────────────

def create_database():
    """Create the Tech database if it does not exist (connects to master)."""
    conn = pyodbc.connect(_connection_string("master"), autocommit=True)
    conn.cursor().execute(
        f"IF DB_ID(N'{SQL_DATABASE}') IS NULL CREATE DATABASE [{SQL_DATABASE}]"
    )
    conn.close()
    print(f"  Database '{SQL_DATABASE}' ready.")


def create_table(conn):
    """Create the Schedule table if it does not exist."""
    conn.cursor().execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Schedule')
        CREATE TABLE dbo.Schedule (
            ScheduleID  INT IDENTITY(1,1) PRIMARY KEY,
            [date]      DATE        NOT NULL,
            [time]      TIME(0)     NOT NULL,
            position    VARCHAR(20) NOT NULL,
            available   BIT         NOT NULL DEFAULT 1
        )
    """)
    conn.commit()
    print("  Table 'dbo.Schedule' ready.")


def seed_data(conn, start: date, end: date):
    """
    Populate the Schedule table with slots for every valid workday
    between start and end dates.
    Mirrors the logic of db_Tech.sql (random availability ~50%).
    """
    rows = []
    current = start
    while current <= end:
        if current.weekday() in VALID_WEEKDAYS:
            for hour in WORK_HOURS:
                hour_obj = dt_time.fromisoformat(hour)
                for position in POSITIONS:
                    available = 1 if random.random() >= 0.5 else 0
                    rows.append((current, hour_obj, position, available))
        current += timedelta(days=1)

    cursor = conn.cursor()
    cursor.fast_executemany = True
    cursor.executemany(
        "INSERT INTO dbo.Schedule ([date], [time], position, available) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    print(f"  Inserted {len(rows)} rows.")


def create_and_seed_db():
    """
    Create the SQL Server DB and populate it with 2026 schedule data.
    Run this ONCE before starting the application.
    """
    print("=" * 50)
    print("Creating SQL Server Database")
    print("=" * 50)

    create_database()

    conn = get_connection()
    create_table(conn)

    existing = conn.cursor().execute("SELECT COUNT(*) FROM dbo.Schedule").fetchone()[0]
    if existing == 0:
        seed_data(conn, date(2026, 1, 1), date(2026, 12, 31))
    else:
        print(f"  Table already has {existing} rows — skipping seed.")
    conn.close()

    print(f"Database ready: {SQL_SERVER}/{SQL_DATABASE}")
    print("=" * 50)


# ── Runtime queries ───────────────────────────────────────────────────────────

def get_available_slots(target_date: str, position: str = "Python Dev", n: int = 3):
    """
    Return the n nearest available slots on or after target_date.

    Parameters
    ----------
    target_date : str   e.g. "2026-03-15"
    position    : str   e.g. "Python Dev"
    n           : int   number of slots to return (default 3)

    Returns
    -------
    list of dicts: [{"date": ..., "time": ..., "ScheduleID": ...}, ...]
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT TOP (?) ScheduleID, [date], [time]
        FROM   dbo.Schedule
        WHERE  [date]    >= ?
          AND  position   = ?
          AND  available  = 1
        ORDER  BY [date], [time]
        """,
        (n, target_date, position),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {"ScheduleID": row[0], "date": str(row[1]), "time": row[2].strftime("%H:%M")}
        for row in rows
    ]


def book_slot(schedule_id: int) -> bool:
    """
    Mark a slot as booked (available = 0).

    Parameters
    ----------
    schedule_id : int   the ScheduleID to book

    Returns
    -------
    bool: True if booking succeeded, False if slot was already taken or on error
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Check the slot is still free
        cursor.execute(
            "SELECT available FROM dbo.Schedule WHERE ScheduleID = ?",
            (schedule_id,),
        )
        row = cursor.fetchone()

        if not row or row[0] == 0:
            return False

        cursor.execute(
            "UPDATE dbo.Schedule SET available = 0 WHERE ScheduleID = ?",
            (schedule_id,),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


# ── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_and_seed_db()

    # Quick test
    print("\nTest — 3 available Python Dev slots from 2026-03-15:")
    slots = get_available_slots("2026-03-15", "Python Dev")
    for slot in slots:
        print(f"  ID={slot['ScheduleID']}  {slot['date']}  {slot['time']}")
