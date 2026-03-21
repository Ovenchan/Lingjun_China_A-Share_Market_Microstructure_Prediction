from __future__ import annotations

import csv
import math
import sqlite3
import tempfile
from itertools import combinations
from pathlib import Path


def _normalize_weights(file_paths: list[str], weights: list[float] | None) -> list[float]:
    if not file_paths:
        raise ValueError("file_paths cannot be empty.")

    if weights is None:
        return [1.0 / len(file_paths)] * len(file_paths)

    if len(weights) != len(file_paths):
        raise ValueError("weights length must match file_paths length.")

    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("sum of weights must not be zero.")

    return [weight / total_weight for weight in weights]


def _import_predictions(conn: sqlite3.Connection, file_paths: list[str]) -> list[int]:
    counts: list[int] = []

    for file_idx, file_path in enumerate(file_paths):
        table_name = f"pred_{file_idx}"
        path = Path(file_path)

        if file_idx == 0:
            conn.execute(
                f"""
                CREATE TABLE {table_name} (
                    seq INTEGER PRIMARY KEY,
                    uid TEXT NOT NULL UNIQUE,
                    prediction REAL NOT NULL
                )
                """
            )
            insert_sql = f"INSERT INTO {table_name} (seq, uid, prediction) VALUES (?, ?, ?)"
        else:
            conn.execute(
                f"""
                CREATE TABLE {table_name} (
                    uid TEXT PRIMARY KEY,
                    prediction REAL NOT NULL
                )
                """
            )
            insert_sql = f"INSERT INTO {table_name} (uid, prediction) VALUES (?, ?)"

        batch = []
        row_count = 0
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["Uid", "prediction"]:
                raise ValueError(f"{file_path} must contain columns: Uid, prediction")

            for seq, row in enumerate(reader):
                row_count += 1
                if file_idx == 0:
                    batch.append((seq, row["Uid"], float(row["prediction"])))
                else:
                    batch.append((row["Uid"], float(row["prediction"])))

                if len(batch) >= 100_000:
                    conn.executemany(insert_sql, batch)
                    batch.clear()

        if batch:
            conn.executemany(insert_sql, batch)

        counts.append(row_count)

    return counts


def _build_join_clause(num_files: int) -> str:
    join_parts = []
    for file_idx in range(1, num_files):
        join_parts.append(f"JOIN pred_{file_idx} USING(uid)")
    return " ".join(join_parts)


def _validate_uid_sets(conn: sqlite3.Connection, num_files: int, base_count: int) -> None:
    join_clause = _build_join_clause(num_files)
    joined_count = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM pred_0
        {join_clause}
        """
    ).fetchone()[0]
    if joined_count != base_count:
        raise ValueError("All files must contain the same Uid set.")


def _pearson_from_sql(conn: sqlite3.Connection, left_idx: int, right_idx: int) -> float:
    cursor = conn.execute(
        f"""
        SELECT pred_{left_idx}.prediction, pred_{right_idx}.prediction
        FROM pred_{left_idx}
        JOIN pred_{right_idx} USING(uid)
        """
    )

    n = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0

    for x, y in cursor:
        n += 1
        sum_x += x
        sum_y += y
        sum_x2 += x * x
        sum_y2 += y * y
        sum_xy += x * y

    numerator = n * sum_xy - sum_x * sum_y
    denominator = (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
    if denominator <= 0:
        return 0.0
    return numerator / math.sqrt(denominator)


def _print_pairwise_correlations(conn: sqlite3.Connection, file_paths: list[str]) -> None:
    print("Pairwise Pearson correlations:")
    for left_idx, right_idx in combinations(range(len(file_paths)), 2):
        corr = _pearson_from_sql(conn, left_idx, right_idx)
        print(
            f"{Path(file_paths[left_idx]).name} vs {Path(file_paths[right_idx]).name}: {corr:.6f}"
        )


def ensemble_submissions(
    file_paths: list[str],
    weights: list[float] | None = None,
    output_path: str = "submission_ensemble.csv",
) -> None:
    normalized_weights = _normalize_weights(file_paths, weights)

    temp_db = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
    temp_db_path = Path(temp_db.name)
    temp_db.close()

    try:
        conn = sqlite3.connect(temp_db_path)
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")

        try:
            counts = _import_predictions(conn, file_paths)
            conn.commit()

            if len(set(counts)) != 1:
                raise ValueError(f"row counts do not match: {counts}")

            _validate_uid_sets(conn, len(file_paths), counts[0])
            _print_pairwise_correlations(conn, file_paths)

            weighted_terms = " + ".join(
                f"pred_{file_idx}.prediction * {weight:.18g}"
                for file_idx, weight in enumerate(normalized_weights)
            )
            join_clause = _build_join_clause(len(file_paths))
            query = f"""
                SELECT pred_0.uid, {weighted_terms} AS prediction
                FROM pred_0
                {join_clause}
                ORDER BY pred_0.seq
            """

            with Path(output_path).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Uid", "prediction"])
                for uid, prediction in conn.execute(query):
                    writer.writerow([uid, prediction])
        finally:
            conn.close()
    finally:
        temp_db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    ensemble_submissions(
        [
            "submission_thgnn.csv",
            # "submission_mlp_time_split.csv",
            "submission_NeurIF.csv",
            "submission_xgb_time_split_001796.csv"
        ],
        [0.35, 0.35, 0.3],
        "submission_ensemble_thgnn_neurif_xgb.csv"
    )
