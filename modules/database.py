"""
Database Module
================
SQLite database for session management, notes, and equation history.
Provides persistent storage for the Smart Notebook application.
"""

import sqlite3
import os
from datetime import datetime


class NotebookDatabase:
    """SQLite database manager for the AI Smart Notebook."""

    def __init__(self, db_path="database/notebook.db"):
        """
        Initialize database connection and create tables.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        self._create_tables()

    def _create_tables(self):
        """Create the database tables if they don't exist."""
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATETIME DEFAULT CURRENT_TIMESTAMP,
                drawing_path TEXT,
                voice_path TEXT
            );

            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                note_text TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS equations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                equation TEXT,
                solution TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)
        self.conn.commit()

    def create_session(self):
        """
        Create a new notebook session.

        Returns:
            The ID of the newly created session.
        """
        self.cursor.execute(
            "INSERT INTO sessions (date) VALUES (?)",
            (datetime.now().isoformat(),)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def save_drawing(self, session_id, filepath):
        """
        Update the drawing path for a session.

        Args:
            session_id: The session ID to update.
            filepath: Path to the saved drawing image.
        """
        self.cursor.execute(
            "UPDATE sessions SET drawing_path = ? WHERE id = ?",
            (filepath, session_id)
        )
        self.conn.commit()

    def save_note(self, session_id, note_text):
        """
        Save a text note to a session.

        Args:
            session_id: The session ID.
            note_text: The note content.

        Returns:
            The ID of the saved note.
        """
        self.cursor.execute(
            "INSERT INTO notes (session_id, note_text) VALUES (?, ?)",
            (session_id, note_text)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def save_equation(self, session_id, equation, solution):
        """
        Save an equation and its solution.

        Args:
            session_id: The session ID.
            equation: The equation string.
            solution: The computed solution string.

        Returns:
            The ID of the saved equation.
        """
        self.cursor.execute(
            "INSERT INTO equations (session_id, equation, solution) VALUES (?, ?, ?)",
            (session_id, equation, solution)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_sessions(self):
        """
        Retrieve all sessions, ordered by most recent first.

        Returns:
            List of session dictionaries.
        """
        self.cursor.execute(
            "SELECT * FROM sessions ORDER BY date DESC"
        )
        return [dict(row) for row in self.cursor.fetchall()]

    def get_session(self, session_id):
        """
        Retrieve a single session by ID.

        Args:
            session_id: The session ID.

        Returns:
            Session dictionary, or None.
        """
        self.cursor.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def get_notes(self, session_id):
        """
        Retrieve all notes for a session.

        Args:
            session_id: The session ID.

        Returns:
            List of note dictionaries.
        """
        self.cursor.execute(
            "SELECT * FROM notes WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        )
        return [dict(row) for row in self.cursor.fetchall()]

    def get_equations(self, session_id):
        """
        Retrieve all equations for a session.

        Args:
            session_id: The session ID.

        Returns:
            List of equation dictionaries.
        """
        self.cursor.execute(
            "SELECT * FROM equations WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        )
        return [dict(row) for row in self.cursor.fetchall()]

    def delete_session(self, session_id):
        """
        Delete a session and all associated data.

        Args:
            session_id: The session ID to delete.
        """
        self.cursor.execute("DELETE FROM notes WHERE session_id = ?", (session_id,))
        self.cursor.execute("DELETE FROM equations WHERE session_id = ?", (session_id,))
        self.cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()
