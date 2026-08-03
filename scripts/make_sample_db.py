#!/usr/bin/env python3
"""Build the offline SQLite fixture and its labeled question set.

Delta's real benchmark is Spider, which is a 206 MB download. This fixture
exists so that the harness, the agent loop, and CI can all run end to end with
no download and no API key. It is a university schema, deliberately shaped like
Spider's databases: five related tables, a mix of text and numeric columns, and
enough rows that aggregates and joins produce non-trivial answers.

The 15 questions span the same difficulty range Spider uses, from single-table
filters up to multi-join aggregates with subqueries, so a weak prompt fails some
of them and a good prompt fails fewer. Run this before anything else:

    python scripts/make_sample_db.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "sample.db"
QUESTIONS_PATH = DATA_DIR / "sample_questions.json"

SCHEMA = """
CREATE TABLE departments (
    dept_id     INTEGER PRIMARY KEY,
    dept_name   TEXT NOT NULL,
    building    TEXT,
    budget      REAL
);

CREATE TABLE instructors (
    inst_id     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    dept_id     INTEGER,
    salary      REAL,
    hire_date   TEXT,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

CREATE TABLE students (
    student_id      INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    dept_id         INTEGER,
    enrollment_year INTEGER,
    gpa             REAL,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

CREATE TABLE courses (
    course_id   INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    dept_id     INTEGER,
    credits     INTEGER,
    inst_id     INTEGER,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id),
    FOREIGN KEY (inst_id) REFERENCES instructors(inst_id)
);

CREATE TABLE enrollments (
    student_id  INTEGER,
    course_id   INTEGER,
    semester    TEXT,
    grade       REAL,
    PRIMARY KEY (student_id, course_id, semester),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);
"""

DEPARTMENTS = [
    (1, "Computer Science", "Turing Hall", 1_850_000.0),
    (2, "Mathematics", "Noether Hall", 940_000.0),
    (3, "Physics", "Bohr Hall", 1_320_000.0),
    (4, "History", "Herodotus Hall", 410_000.0),
    (5, "Biology", "Darwin Hall", 1_105_000.0),
]

INSTRUCTORS = [
    (1, "Ada Chen", 1, 145_000.0, "2011-08-15"),
    (2, "Marcus Webb", 1, 132_500.0, "2015-01-10"),
    (3, "Priya Raman", 1, 158_000.0, "2008-09-01"),
    (4, "Elena Voss", 2, 121_000.0, "2013-08-20"),
    (5, "Tomas Lindqvist", 2, 99_500.0, "2019-01-07"),
    (6, "Grace Okonkwo", 3, 137_000.0, "2010-08-16"),
    (7, "Hiroshi Tanaka", 3, 118_000.0, "2017-08-21"),
    (8, "Rosa Delgado", 4, 88_000.0, "2012-01-09"),
    (9, "Samuel Bright", 5, 126_500.0, "2014-08-18"),
    (10, "Nadia Farouk", 5, 141_000.0, "2009-08-17"),
    # These two teach no courses. Without them, "instructors who teach nothing"
    # returns an empty set, which any broken query would trivially satisfy.
    (11, "Yusuf Demir", 4, 79_500.0, "2021-01-11"),
    (12, "Clara Nkemelu", 2, 104_000.0, "2020-08-19"),
]

STUDENTS = [
    (1, "Liam Foster", 1, 2021, 3.82),
    (2, "Zoe Nakamura", 1, 2021, 3.45),
    (3, "Omar Haddad", 1, 2022, 3.91),
    (4, "Chloe Bennett", 1, 2023, 2.78),
    (5, "Ravi Patel", 2, 2021, 3.66),
    (6, "Ingrid Larsen", 2, 2022, 3.12),
    (7, "Diego Morales", 3, 2020, 3.55),
    (8, "Fatima Nasser", 3, 2022, 3.98),
    (9, "Henry Whitfield", 4, 2021, 2.95),
    (10, "Aiko Yamamoto", 5, 2020, 3.74),
    (11, "Lucas Meyer", 5, 2023, 3.28),
    (12, "Sofia Rossi", 5, 2022, 3.61),
]

COURSES = [
    (101, "Introduction to Programming", 1, 4, 1),
    (102, "Data Structures", 1, 4, 1),
    (103, "Operating Systems", 1, 3, 2),
    (104, "Machine Learning", 1, 3, 3),
    (201, "Linear Algebra", 2, 4, 4),
    (202, "Real Analysis", 2, 3, 4),
    (203, "Discrete Mathematics", 2, 3, 5),
    (301, "Classical Mechanics", 3, 4, 6),
    (302, "Quantum Mechanics", 3, 3, 6),
    (303, "Thermodynamics", 3, 3, 7),
    (401, "Ancient Civilizations", 4, 3, 8),
    (501, "Cell Biology", 5, 4, 9),
    (502, "Genetics", 5, 3, 10),
    (503, "Ecology", 5, 3, 10),
]

ENROLLMENTS = [
    (1, 101, "Fall 2021", 3.7),
    (1, 102, "Spring 2022", 4.0),
    (1, 104, "Fall 2023", 3.9),
    (2, 101, "Fall 2021", 3.3),
    (2, 102, "Spring 2022", 3.0),
    (2, 103, "Fall 2022", 3.7),
    (3, 101, "Fall 2022", 4.0),
    (3, 102, "Spring 2023", 4.0),
    (3, 104, "Fall 2023", 3.8),
    (3, 201, "Spring 2023", 3.6),
    (4, 101, "Fall 2023", 2.3),
    (4, 203, "Fall 2023", 2.7),
    (5, 201, "Fall 2021", 3.9),
    (5, 202, "Spring 2022", 3.4),
    (5, 203, "Fall 2022", 3.8),
    (6, 201, "Fall 2022", 3.1),
    (6, 203, "Spring 2023", 3.0),
    (7, 301, "Fall 2020", 3.5),
    (7, 302, "Spring 2021", 3.4),
    (7, 303, "Fall 2021", 3.8),
    (8, 301, "Fall 2022", 4.0),
    (8, 302, "Spring 2023", 3.9),
    (8, 201, "Fall 2022", 3.7),
    (9, 401, "Fall 2021", 2.9),
    (10, 501, "Fall 2020", 3.8),
    (10, 502, "Spring 2021", 3.7),
    (10, 503, "Fall 2021", 3.6),
    (11, 501, "Fall 2023", 3.2),
    (12, 501, "Fall 2022", 3.6),
    (12, 502, "Spring 2023", 3.7),
]

# Each question carries the gold SQL and a difficulty label using Spider's own
# vocabulary. `make test` asserts every gold query executes and self-scores.
QUESTIONS = [
    {
        "id": "s001",
        "question": "How many students are there?",
        "gold": "SELECT count(*) FROM students",
        "difficulty": "easy",
    },
    {
        "id": "s002",
        "question": "List the names of all departments in alphabetical order.",
        "gold": "SELECT dept_name FROM departments ORDER BY dept_name",
        "difficulty": "easy",
    },
    {
        "id": "s003",
        "question": "What are the names of students with a GPA above 3.7?",
        "gold": "SELECT name FROM students WHERE gpa > 3.7",
        "difficulty": "easy",
    },
    {
        "id": "s004",
        "question": "What is the average salary of instructors?",
        "gold": "SELECT avg(salary) FROM instructors",
        "difficulty": "easy",
    },
    {
        "id": "s005",
        "question": "Which building houses the Physics department?",
        "gold": "SELECT building FROM departments WHERE dept_name = 'Physics'",
        "difficulty": "easy",
    },
    {
        "id": "s006",
        "question": "Show the name of each student along with their department name.",
        "gold": (
            "SELECT T1.name, T2.dept_name FROM students AS T1 "
            "JOIN departments AS T2 ON T1.dept_id = T2.dept_id"
        ),
        "difficulty": "medium",
    },
    {
        "id": "s007",
        "question": "How many students are enrolled in each department? Show the department name and the count.",
        "gold": (
            "SELECT T2.dept_name, count(*) FROM students AS T1 "
            "JOIN departments AS T2 ON T1.dept_id = T2.dept_id GROUP BY T2.dept_name"
        ),
        "difficulty": "medium",
    },
    {
        "id": "s008",
        "question": "What is the title of the course with the most enrollments?",
        "gold": (
            "SELECT T2.title FROM enrollments AS T1 JOIN courses AS T2 "
            "ON T1.course_id = T2.course_id GROUP BY T1.course_id "
            "ORDER BY count(*) DESC LIMIT 1"
        ),
        "difficulty": "medium",
    },
    {
        "id": "s009",
        "question": "Find the names of the three students with the highest GPA, highest first.",
        "gold": "SELECT name FROM students ORDER BY gpa DESC LIMIT 3",
        "difficulty": "medium",
    },
    {
        "id": "s010",
        "question": "Which departments have a budget greater than one million? Give their names and budgets.",
        "gold": "SELECT dept_name, budget FROM departments WHERE budget > 1000000",
        "difficulty": "medium",
    },
    {
        "id": "s011",
        "question": "Show the names of departments that have more than two instructors.",
        "gold": (
            "SELECT T2.dept_name FROM instructors AS T1 JOIN departments AS T2 "
            "ON T1.dept_id = T2.dept_id GROUP BY T2.dept_name HAVING count(*) > 2"
        ),
        "difficulty": "hard",
    },
    {
        "id": "s012",
        "question": "What are the names of students whose GPA is above the average GPA of all students?",
        "gold": "SELECT name FROM students WHERE gpa > (SELECT avg(gpa) FROM students)",
        "difficulty": "hard",
    },
    {
        "id": "s013",
        "question": "List the names of instructors who do not teach any course.",
        "gold": (
            "SELECT name FROM instructors WHERE inst_id NOT IN "
            "(SELECT inst_id FROM courses WHERE inst_id IS NOT NULL)"
        ),
        "difficulty": "hard",
    },
    {
        "id": "s014",
        "question": (
            "For each department, show the department name and the average grade its students "
            "have earned, but only for departments where that average is above 3.5."
        ),
        "gold": (
            "SELECT T3.dept_name, avg(T1.grade) FROM enrollments AS T1 "
            "JOIN students AS T2 ON T1.student_id = T2.student_id "
            "JOIN departments AS T3 ON T2.dept_id = T3.dept_id "
            "GROUP BY T3.dept_name HAVING avg(T1.grade) > 3.5"
        ),
        "difficulty": "extra",
    },
    {
        "id": "s015",
        "question": (
            "Show the name of each instructor and the number of distinct students enrolled in "
            "the courses they teach, ordered by that count from highest to lowest."
        ),
        "gold": (
            "SELECT T1.name, count(DISTINCT T3.student_id) FROM instructors AS T1 "
            "JOIN courses AS T2 ON T1.inst_id = T2.inst_id "
            "JOIN enrollments AS T3 ON T2.course_id = T3.course_id "
            "GROUP BY T1.name ORDER BY count(DISTINCT T3.student_id) DESC"
        ),
        "difficulty": "extra",
    },
]


def build_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", DEPARTMENTS)
        conn.executemany("INSERT INTO instructors VALUES (?, ?, ?, ?, ?)", INSTRUCTORS)
        conn.executemany("INSERT INTO students VALUES (?, ?, ?, ?, ?)", STUDENTS)
        conn.executemany("INSERT INTO courses VALUES (?, ?, ?, ?, ?)", COURSES)
        conn.executemany("INSERT INTO enrollments VALUES (?, ?, ?, ?)", ENROLLMENTS)
        conn.commit()
    finally:
        conn.close()


def write_questions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{**q, "db_id": "sample"} for q in QUESTIONS]
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    build_database(DB_PATH)
    write_questions(QUESTIONS_PATH)
    counts = {
        "departments": len(DEPARTMENTS),
        "instructors": len(INSTRUCTORS),
        "students": len(STUDENTS),
        "courses": len(COURSES),
        "enrollments": len(ENROLLMENTS),
    }
    print(f"wrote {DB_PATH}")
    for table, n in counts.items():
        print(f"  {table:<14} {n} rows")
    print(f"wrote {QUESTIONS_PATH} ({len(QUESTIONS)} labeled questions)")


if __name__ == "__main__":
    main()
