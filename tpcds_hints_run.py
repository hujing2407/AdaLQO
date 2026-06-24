import os
import json
import time
import hashlib
import re
import psycopg2
import pandas as pd

from config import Config

# =========================================================
# CONFIG
# =========================================================

logger = Config.setup_logging()
HINTS_OFF = Config.HINTS_OFF

db_params = Config.DB_CONFIG
dbname = db_params["dbname"]

# QUERY_FILE = "dataset/queries/tpcds_queries/tpcds_10gb_queries_1to8j_0.25addjp_0.5lp_10k.sql"
QUERY_FILE = "dataset/queries/tpcds_queries/tpcds_test.sql"
RESULT_DIR = "dataset/tpc-ds"

PLAN_DIR = os.path.join(RESULT_DIR, "plans")
EXECUTE_DIR = os.path.join(RESULT_DIR, "execute_json")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLAN_DIR, exist_ok=True)
os.makedirs(EXECUTE_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULT_DIR, "execution_results.csv")
JSONL_PATH = os.path.join(RESULT_DIR, "execution_results.jsonl")

STATEMENT_TIMEOUT_MS = 60_000

DEFAULT_EXPLAIN_OPTIONS = "ANALYZE, BUFFERS, VERBOSE, FORMAT JSON"

QUERY_CANCELED_SQLSTATE = "57014"
UNDEFINED_FUNCTION_SQLSTATE = "42883"


# =========================================================
# UTILS
# =========================================================

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_sql_for_hash(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip())


def strip_final_semicolon(sql: str) -> str:
    s = sql.strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def make_query_id(query_index: int, sql: str) -> str:
    sql_hash = sha1_text(normalize_sql_for_hash(sql))
    return f"q{query_index:05d}_{sql_hash[:10]}"


def iter_queries_line_by_line(query_file: str):
    """
    你的文件格式：
        line 1: SELECT ...
        line 2:
        line 3: SELECT ...
        line 4:

    每个非空行是一条 Query。
    """
    query_index = 0

    with open(query_file, "r", encoding="utf-8", errors="replace") as f:
        for source_line_no, line in enumerate(f, start=1):
            sql = line.strip()

            if not sql:
                continue

            query_index += 1

            sql = strip_final_semicolon(sql)
            sql_hash = sha1_text(normalize_sql_for_hash(sql))
            query_id = make_query_id(query_index, sql)

            yield {
                "query_index": query_index,
                "query_id": query_id,
                "sql_hash": sql_hash,
                "source_line_no": source_line_no,
                "sql": sql,
            }


def append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()


def reset_pg_settings(cur):
    cur.execute("RESET ALL;")


def apply_timeout(cur):
    cur.execute(f"SET statement_timeout = {int(STATEMENT_TIMEOUT_MS)};")


def apply_hint(cur, hint_sql: str):
    if hint_sql.strip():
        cur.execute(hint_sql)


def make_explain_sql(sql: str) -> str:
    sql = strip_final_semicolon(sql)
    return f"""
EXPLAIN ({DEFAULT_EXPLAIN_OPTIONS})
{sql}
"""


def sum_plan_buffers(plan: dict) -> dict:
    """
    递归统计整个 plan tree 的 buffer 信息。
    PostgreSQL 的 buffer 字段通常在每个 Plan node 里，
    不是只在 root result 里。
    """
    keys = [
        "Shared Hit Blocks",
        "Shared Read Blocks",
        "Shared Dirtied Blocks",
        "Shared Written Blocks",
        "Local Hit Blocks",
        "Local Read Blocks",
        "Local Dirtied Blocks",
        "Local Written Blocks",
        "Temp Read Blocks",
        "Temp Written Blocks",
    ]

    total = {k: 0 for k in keys}

    def visit(node):
        if not isinstance(node, dict):
            return

        for k in keys:
            v = node.get(k)
            if isinstance(v, (int, float)):
                total[k] += v

        for child in node.get("Plans", []):
            visit(child)

    visit(plan)
    return total


def classify_error(e: Exception) -> str:
    pgcode = getattr(e, "pgcode", None)
    msg = str(e).lower()

    if pgcode == QUERY_CANCELED_SQLSTATE:
        return "timeout"

    if pgcode == UNDEFINED_FUNCTION_SQLSTATE and "operator does not exist" in msg:
        return "schema_type_error"

    return "error"


# =========================================================
# RUN QUERY
# =========================================================

def run_query(cur, sql: str):
    explain_sql = make_explain_sql(sql)

    start = time.time()
    cur.execute(explain_sql)
    wallclock_ms = (time.time() - start) * 1000.0

    full_explain_json = cur.fetchone()[0][0]

    plan = full_explain_json.get("Plan")
    planning_time = full_explain_json.get("Planning Time")
    execution_time = full_explain_json.get("Execution Time")

    buffers = sum_plan_buffers(plan)

    buffer_hit = buffers.get("Shared Hit Blocks")
    buffer_read = buffers.get("Shared Read Blocks")

    return {
        "full_explain_json": full_explain_json,
        "plan": plan,
        "planning_time_ms": planning_time,
        "execution_time_ms": execution_time,
        "wallclock_ms": wallclock_ms,
        "buffers": buffers,
        "buffer_hit": buffer_hit,
        "buffer_read": buffer_read,
    }


# =========================================================
# POSTGRES CONNECTION
# =========================================================

conn = psycopg2.connect(**db_params)
conn.autocommit = True
cursor = conn.cursor()

# =========================================================
# EXECUTE ALL QUERIES
# =========================================================

records = []

try:
    for idx, item in enumerate(iter_queries_line_by_line(QUERY_FILE), start=1):
        sql = item["sql"]
        query_id = item["query_id"]

        plan_list = []
        latency_list = []
        planning_time_list = []
        wallclock_ms_list = []
        buffer_hit_list = []
        buffer_read_list = []
        buffer_total_list = []
        status_list = []
        error_list = []
        full_explain_list = []

        logger.info(
            f"Executing query_id={query_id}, "
            f"query_index={item['query_index']}, "
            f"source_line_no={item['source_line_no']}"
        )

        default_failed = False
        default_failure_type = None

        for hint_id, hint_sql in enumerate(HINTS_OFF):
            if default_failed:
                plan_list.append(None)
                latency_list.append(float("inf"))
                planning_time_list.append(None)
                wallclock_ms_list.append(None)
                buffer_hit_list.append(None)
                buffer_read_list.append(None)
                buffer_total_list.append(None)
                status_list.append(f"skipped_due_default_{default_failure_type}")
                error_list.append(f"hint_id=0 failed with {default_failure_type}")
                full_explain_list.append(None)
                continue

            try:
                reset_pg_settings(cursor)
                apply_timeout(cursor)
                apply_hint(cursor, hint_sql)

                logger.info(
                    f"Executing query_id={query_id} with hint_id={hint_id}"
                )

                result = run_query(cursor, sql)

                plan_list.append(result["plan"])
                latency_list.append(result["execution_time_ms"])
                planning_time_list.append(result["planning_time_ms"])
                wallclock_ms_list.append(result["wallclock_ms"])
                buffer_hit_list.append(result["buffer_hit"])
                buffer_read_list.append(result["buffer_read"])
                buffer_total_list.append(result["buffers"])
                status_list.append("ok")
                error_list.append(None)
                full_explain_list.append(result["full_explain_json"])

                logger.info(
                    f"OK query_id={query_id}, hint_id={hint_id}, "
                    f"latency_ms={result['execution_time_ms']}, "
                    f"wallclock_ms={result['wallclock_ms']:.2f}"
                )

            except Exception as e:
                err_type = classify_error(e)

                logger.error(
                    f"Query failed: query_id={query_id}, hint_id={hint_id}, error_type={err_type}"
                )
                logger.error(e)

                plan_list.append(None)
                latency_list.append(float("inf"))
                planning_time_list.append(None)
                wallclock_ms_list.append(None)
                buffer_hit_list.append(None)
                buffer_read_list.append(None)
                buffer_total_list.append(None)
                status_list.append(err_type)
                error_list.append(str(e))
                full_explain_list.append(None)

                if hint_id == 0 and err_type in {"timeout", "schema_type_error"}:
                    default_failed = True
                    default_failure_type = err_type

        best_execute_time = min(latency_list)

        plan_path = os.path.join(
            PLAN_DIR,
            f"{query_id}_plans.json"
        )

        execute_path = os.path.join(
            EXECUTE_DIR,
            f"{query_id}_execute.json"
        )

        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_list, f, indent=2, ensure_ascii=False, default=str)

        execute_record = {
            "dbname": dbname,
            "query_index": item["query_index"],
            "query_id": item["query_id"],
            "sql_hash": item["sql_hash"],
            "source_line_no": item["source_line_no"],
            "sql": sql,
            "num_hints": len(HINTS_OFF),
            "hints": HINTS_OFF,
            "status_list": status_list,
            "latency_list": latency_list,
            "planning_time_list": planning_time_list,
            "wallclock_ms_list": wallclock_ms_list,
            "buffer_hit_list": buffer_hit_list,
            "buffer_read_list": buffer_read_list,
            "buffer_total_list": buffer_total_list,
            "error_list": error_list,
            "plan_list": plan_list,
            "full_explain_list": full_explain_list,
            "best_execute_time": best_execute_time,
        }

        with open(execute_path, "w", encoding="utf-8") as f:
            json.dump(execute_record, f, indent=2, ensure_ascii=False, default=str)

        record = {
            "dbname": dbname,
            "query_index": item["query_index"],
            "query_id": item["query_id"],
            "sql_hash": item["sql_hash"],
            "source_line_no": item["source_line_no"],
            "latency_list": latency_list,
            "planning_time_list": planning_time_list,
            "wallclock_ms_list": wallclock_ms_list,
            "buffer_hit_list": buffer_hit_list,
            "buffer_read_list": buffer_read_list,
            "status_list": status_list,
            "error_list": error_list,
            "best_execute_time": best_execute_time,
            "plan_path": plan_path,
            "execute_path": execute_path,
        }

        records.append(record)

        append_jsonl(JSONL_PATH, record)

        if idx % 10 == 0:
            logger.info(f"Executed {idx} queries")
            df = pd.DataFrame(records)
            df.to_csv(CSV_PATH, index=False)

except Exception as e:
    logger.error("Fatal error while executing queries.")
    logger.error(e)

finally:
    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False)

    cursor.close()
    conn.close()

    logger.info("DONE executing queries.")
    logger.info(f"CSV saved to: {CSV_PATH}")
    logger.info(f"JSONL saved to: {JSONL_PATH}")
