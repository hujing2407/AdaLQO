import os
import json
import time
import psycopg2
import sys
from config import Config
import pandas as pd

logger = Config.setup_logging()
HINTS_OFF  = Config.HINTS_OFF
# =========================================================
# SETUP DB CONNECTION & LOAD METADATA
# =========================================================
db_params = Config.get_db_params()
dbname = db_params["dbname"]
QUERY_DIR = "dataset/queries/tpc-h_sf1_10"
RESULT_DIR = "dataset/tpc-h_sf1_10_buf"
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
records = []
for idx, item in enumerate(metadata):
    with open(item["sql_path"], "r") as f:
        sql = f.read()
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
                logger.info(f"Executing query: {item['query_id']} with {i}th plan in {item['phase']}.")
                plan_json, latency, buffer_info = run_query(cursor, sql)

                plan_list.append(plan_json)
                latency_list.append(latency)
                buffer_hit_list.append(buffer_info["shared_hit_blocks"])
                buffer_read_list.append(buffer_info["shared_read_blocks"])
                buffer_info_list.append(buffer_info)
            except Exception as e:
                logger.error(f"Query failed: {sql} with {i}th plan.")
                logger.error(e)
                plan_list.append(None)
                latency_list.append(float("inf"))
                buffer_hit_list.append(None)
                buffer_read_list.append(None)
                buffer_info_list.append(None)

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
            "buffer_info_list": buffer_info_list,
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
