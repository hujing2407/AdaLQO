import os
import json
import time
import psycopg2
import sys
from config import Config
import pandas as pd

logger = Config.setup_logging()
HINTS_OFF  = ['',
             'set enable_nestloop = off;',
             'set enable_nestloop = off; set enable_indexscan = off;',
             'set enable_hashjoin = off;',
             'set enable_hashjoin = off; set enable_indexscan = off;',
             'set enable_mergejoin = off;',
             'set enable_mergejoin = off; set enable_indexscan = off;',
             'set enable_nestloop = off; set enable_mergejoin = off;',
             'set enable_nestloop = off; set enable_mergejoin = off; set enable_indexscan = off;',
             'set enable_nestloop = off; set enable_hashjoin = off;',
             'set enable_nestloop = off; set enable_hashjoin = off; set enable_indexscan = off;',
             'set enable_mergejoin = off; set enable_hashjoin = off;',
             'set enable_mergejoin = off; set enable_hashjoin = off; set enable_indexscan = off;']
# =========================================================
# SETUP DB CONNECTION & LOAD METADATA
# =========================================================
db_params = Config.get_db_params()
dbname = db_params["dbname"]
QUERY_DIR = "AdaLQO/dataset/queries/tpc-h"
RESULT_DIR = "dataset/tpc-h"
os.makedirs(RESULT_DIR, exist_ok=True)

metadata_path = os.path.join(
    QUERY_DIR,
    "metadata.json"
)
with open(metadata_path, "r") as f:
    metadata = json.load(f)

def reset_pg_settings(cur):
    cur.execute("RESET ALL;")

# =========================================================
# POSTGRES CONNECTION
# =========================================================
conn = psycopg2.connect(**db_params)
conn.autocommit = True
cursor = conn.cursor()

# =========================================================
# RUN QUERY
# =========================================================


def run_query(cur,sql):
    explain_sql = f"""
    EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
    {sql}
    """
    start = time.time()
    cur.execute(explain_sql)
    # wallclock = time.time() - start
    result = cursor.fetchone()[0][0]
    execution_time = result["Execution Time"]
    plan = result["Plan"]
    buffer_hit = result.get("Shared Hit Blocks")
    buffer_read = result.get("Shared Read Blocks")

    # actual_rows = plan.get("Actual Rows", None)
    # estimated_rows = plan.get("Plan Rows", None)
    return plan, execution_time, buffer_hit, buffer_read

# =========================================================
# EXECUTE ALL QUERIES
# =========================================================
records = []
for idx, item in enumerate(metadata):
    with open(item["sql_path"], "r") as f:
        sql = f.read()
    try:
        plan_list = []
        latency_list = []
        buffer_hit_list = []
        buffer_read_list = []
        for i, hint_sql in enumerate(HINTS_OFF):
            reset_pg_settings(cursor)
            if hint_sql.strip():
                cursor.execute(hint_sql)
            try:
                logger.info(f"Executing query: {item['query_id']} with {i}th plan in {item['phase']}.")
                plan_json, latency, buffer_hit, buffer_read = run_query(cursor, sql)
                plan_list.append(plan_json)
                latency_list.append(latency)
                buffer_hit_list.append(buffer_hit)
                buffer_read_list.append(buffer_read)
            except Exception as e:
                logger.error(f"Query failed: {sql} with {i}th plan.")
                logger.error(e)
                plan_list.append(None)
                latency_list.append(float("inf"))
                buffer_hit_list.append(None)
                buffer_read_list.append(None)

        best_execute_time = min(latency_list)
        phase_dir = os.path.join(
            RESULT_DIR,
            item["phase"],
            item["query_template"]
        )
        os.makedirs(phase_dir, exist_ok=True)

        # Save plan
        plan_path = os.path.join(
            phase_dir,
            f"{item['query_id']}_plans.json"
        )

        with open(plan_path, "w") as f:
            json.dump(plan_list, f, indent=2)

        records.append({
            "phase": item["phase"],
            "drift_level": item["drift_level"],
            "query_template": item["query_template"],
            "query_id": item["query_id"],
            "latency_list": latency_list,
            "buffer_hit_list": buffer_hit_list,
            "buffer_read_list": buffer_read_list,
            "plan_path": plan_path
        })

        if idx % 10 == 0:
            logger.info (f"Executed {idx}/{len(metadata)}")

    except Exception as e:
        logger.error(f"\nERROR: {item['query_id']}")
        logger.error(e)

# =========================================================
# SAVE RESULTS
# =========================================================
df = pd.DataFrame(records)
csv_path = os.path.join(
    RESULT_DIR,
    "execution_results.csv"
)
df.to_csv(csv_path, index=False)
logger.info("\nDONE executing queries.")

cursor.close()
conn.close()
