import os
import json
import time
import psycopg2
import sys
from config import Config
import pandas as pd

logger = Config.setup_logging()
HINTS_OFF = Config.HINTS_OFF
# =========================================================
# SETUP DB CONNECTION & LOAD QUERIES
# =========================================================
db_params = Config.get_db_params()
dbname = db_params["dbname"]
QUERY_DIR = "dataset/queries/tpcds_queries"
RESULT_DIR = "dataset/tpc-ds"
os.makedirs(RESULT_DIR, exist_ok=True)

sql_path = os.path.join(QUERY_DIR, "tpcds_10gb_queries_1to8j_0.25addjp_0.5lp_10k.sql")


def reset_pg_settings(cur):
    cur.execute("RESET ALL;")


conn = psycopg2.connect(**db_params)
conn.autocommit = True
cursor = conn.cursor()


# =========================================================
# RUN QUERY
# =========================================================
def collect_buffer_info(plan):
    """
    Recursively collect buffer statistics from PostgreSQL EXPLAIN JSON plan tree.
    """
    stats = {
        "shared_hit_blocks": 0,
        "shared_read_blocks": 0,
        "shared_dirtied_blocks": 0,
        "shared_written_blocks": 0,
        "local_hit_blocks": 0,
        "local_read_blocks": 0,
        "temp_read_blocks": 0,
        "temp_written_blocks": 0,
    }

    def visit(node):
        stats["shared_hit_blocks"] += node.get("Shared Hit Blocks", 0)
        stats["shared_read_blocks"] += node.get("Shared Read Blocks", 0)
        stats["shared_dirtied_blocks"] += node.get("Shared Dirtied Blocks", 0)
        stats["shared_written_blocks"] += node.get("Shared Written Blocks", 0)
        stats["local_hit_blocks"] += node.get("Local Hit Blocks", 0)
        stats["local_read_blocks"] += node.get("Local Read Blocks", 0)
        stats["temp_read_blocks"] += node.get("Temp Read Blocks", 0)
        stats["temp_written_blocks"] += node.get("Temp Written Blocks", 0)

        for child in node.get("Plans", []):
            visit(child)

    visit(plan)
    return stats


def run_query(cur, sql):
    explain_sql = f"""
    EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
    {sql}
    """
    cur.execute(explain_sql)

    result = cur.fetchone()[0][0]

    execution_time = result["Execution Time"]
    plan = result["Plan"]

    buffer_info = collect_buffer_info(plan)

    return plan, execution_time, buffer_info


# =========================================================
# EXECUTE ALL QUERIES
# =========================================================
counter = 0
query_ids_set = set()
result_path = os.path.join(RESULT_DIR, "execution_results.csv")
if os.path.exists(result_path):
    df = pd.read_csv(result_path)
    query_ids_set = set(df["query_id"])

with open(sql_path, 'r', encoding='utf-8') as queries:
    for _, q in enumerate(queries):
        q = q.rstrip('\n')
        if not q: continue

        counter += 1
        if counter in query_ids_set: continue  # using a set to check if it's executed

        try:
            plan_list = []
            latency_list = []
            buffer_hit_list = []
            buffer_read_list = []
            buffer_info_list = []
            for i, hint_sql in enumerate(HINTS_OFF):
                reset_pg_settings(cursor)
                if hint_sql.strip():
                    cursor.execute(hint_sql)
                try:
                    logger.info(f"Executing query: {counter} with {i}th plan.")
                    plan_json, latency, buffer_info = run_query(cursor, q)

                    plan_list.append(plan_json)
                    latency_list.append(latency)
                    buffer_hit_list.append(buffer_info["shared_hit_blocks"])
                    buffer_read_list.append(buffer_info["shared_read_blocks"])
                    buffer_info_list.append(buffer_info)
                except Exception as e:
                    logger.error(f"Query failed: {q} with {i}th plan.")
                    logger.error(e)
                    plan_list.append(None)
                    latency_list.append(float("inf"))
                    buffer_hit_list.append(None)
                    buffer_read_list.append(None)
                    buffer_info_list.append(None)

            plans_dir = os.path.join(RESULT_DIR, "plans")
            os.makedirs(plans_dir, exist_ok=True)

            # Save plan
            plan_path = os.path.join(
                plans_dir,
                f"{counter}_plans.json"
            )

            with open(plan_path, "w") as f:
                json.dump(plan_list, f, indent=2)

            record = {
                "query_id": counter,
                "latency_list": latency_list,
                "buffer_hit_list": buffer_hit_list,
                "buffer_read_list": buffer_read_list,
                "buffer_info_list": buffer_info_list,
                "plan_path": plan_path
            }

        except Exception as e:
            logger.error(f"\nERROR: No. {counter} query execution error.")
            logger.error(e)

        # =========================================================
        # SAVE RESULTS
        # =========================================================
        df = pd.DataFrame([record])
        needs_header = not os.path.exists(result_path) or os.path.getsize(result_path) == 0
        df.to_csv(result_path, mode='a', index=False, header=needs_header)

logger.info("\nDONE executing queries.")

cursor.close()
conn.close()
