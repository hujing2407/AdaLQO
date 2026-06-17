import os
import json
import random
from datetime import datetime, timedelta

# =========================================================
# CONFIG
# =========================================================
OUTPUT_DIR = "dataset/queries/tpc-h_sf1_100"
NUM_INSTANCES_PER_QUERY = 100
random.seed(42)

# =========================================================
# CREATE DIR
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# QUERY TEMPLATES
# =========================================================
Q3_TEMPLATE = """
SELECT
    l.l_orderkey,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
    o.o_orderdate,
    o.o_shippriority
FROM customer c
JOIN orders o
    ON c.c_custkey = o.o_custkey
JOIN lineitem l
    ON l.l_orderkey = o.o_orderkey
WHERE
    c.c_mktsegment = '{mktsegment}'
    AND o.o_orderdate < DATE '{orderdate}'
    AND l.l_shipdate > DATE '{shipdate}'
GROUP BY
    l.l_orderkey,
    o.o_orderdate,
    o.o_shippriority
LIMIT 10;
"""

Q5_TEMPLATE = """
SELECT
    n.n_name,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
FROM customer c
JOIN orders o
    ON c.c_custkey = o.o_custkey
JOIN lineitem l
    ON l.l_orderkey = o.o_orderkey
JOIN supplier s
    ON l.l_suppkey = s.s_suppkey
JOIN nation n
    ON s.s_nationkey = n.n_nationkey
JOIN region r
    ON n.n_regionkey = r.r_regionkey
WHERE
    r.r_name = '{region}'
    AND o.o_orderdate >= DATE '{start_date}'
    AND o.o_orderdate < DATE '{end_date}'
GROUP BY
    n.n_name;
"""

Q9_TEMPLATE = """
SELECT
    n.n_name,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS profit
FROM part p
JOIN partsupp ps
    ON p.p_partkey = ps.ps_partkey
JOIN supplier s
    ON ps.ps_suppkey = s.s_suppkey
JOIN nation n
    ON s.s_nationkey = n.n_nationkey
JOIN lineitem l
    ON l.l_partkey = p.p_partkey
    AND l.l_suppkey = s.s_suppkey
JOIN orders o
    ON l.l_orderkey = o.o_orderkey
WHERE
    p.p_name LIKE '%{color}%'
GROUP BY
    n.n_name;
"""

Q21_TEMPLATE = """
SELECT
    s.s_name,
    COUNT(*) AS numwait
FROM supplier s
JOIN lineitem l1
    ON s.s_suppkey = l1.l_suppkey
JOIN orders o
    ON o.o_orderkey = l1.l_orderkey
JOIN nation n
    ON s.s_nationkey = n.n_nationkey
WHERE
    o.o_orderstatus = 'F'
    AND l1.l_receiptdate > l1.l_commitdate
    AND n.n_name = '{nation}'
GROUP BY
    s.s_name
LIMIT 100;
"""

# =========================================================
# PARAMETER SPACES
# =========================================================

MKTSEGMENTS = [
    "AUTOMOBILE",
    "BUILDING",
    "FURNITURE",
    "HOUSEHOLD",
    "MACHINERY"
]

REGIONS = [
    "AFRICA",
    "AMERICA",
    "ASIA",
    "EUROPE",
    "MIDDLE EAST"
]

NATIONS = [
    "CANADA",
    "UNITED STATES",
    "GERMANY",
    "FRANCE",
    "JAPAN",
    "CHINA"
]

COLORS = [
    "green",
    "red",
    "blue",
    "yellow",
    "black"
]

# =========================================================
# DRIFT
# =========================================================

def generate_date(drift_level):
    base_date = datetime(1992, 1, 1)
    if drift_level == "mild":
        offset = random.randint(0, 1000)
    elif drift_level == "moderate":
        offset = random.randint(1000, 2000)
    else:
        offset = random.randint(2000, 2500)
    dt = base_date + timedelta(days=offset)
    return dt.strftime("%Y-%m-%d")

# =========================================================
# QUERY GENERATORS
# =========================================================

def generate_q3(drift_level):
    return Q3_TEMPLATE.format(
        mktsegment=random.choice(MKTSEGMENTS),
        orderdate=generate_date(drift_level),
        shipdate=generate_date(drift_level)
    )

def generate_q5(drift_level):
    start_date = generate_date(drift_level)
    dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = (dt + timedelta(days=365)).strftime("%Y-%m-%d")

    return Q5_TEMPLATE.format(
        region=random.choice(REGIONS),
        start_date=start_date,
        end_date=end_date
    )

def generate_q9(drift_level):
    return Q9_TEMPLATE.format(
        color=random.choice(COLORS)
    )

def generate_q21(drift_level):

    return Q21_TEMPLATE.format(
        nation=random.choice(NATIONS)
    )

QUERY_GENERATORS = {
    "Q3": generate_q3,
    "Q5": generate_q5,
    "Q9": generate_q9,
    "Q21": generate_q21
}

# =========================================================
# PHASES
# =========================================================

PHASES = [
    ("phase_0", "mild"),
    ("phase_1", "moderate"),
    ("phase_2", "severe")
]

# =========================================================
# GENERATE
# =========================================================

metadata = []

for phase_name, drift_level in PHASES:
    phase_dir = os.path.join(OUTPUT_DIR, phase_name)
    os.makedirs(phase_dir, exist_ok=True)
    for query_name, generator in QUERY_GENERATORS.items():
        query_dir = os.path.join(phase_dir, query_name)
        os.makedirs(query_dir, exist_ok=True)
        for i in range(NUM_INSTANCES_PER_QUERY):
            sql = generator(drift_level)
            query_id = f"{query_name}_{i}"
            sql_path = os.path.join(
                query_dir,
                f"{query_id}.sql"
            )

            with open(sql_path, "w") as f:
                f.write(sql)

            metadata.append({
                "phase": phase_name,
                "drift_level": drift_level,
                "query_template": query_name,
                "query_id": query_id,
                "sql_path": sql_path
            })

            if i % 100 == 0:
                print(f"{phase_name} {query_name} {i}")

# =========================================================
# SAVE METADATA
# =========================================================
metadata_path = os.path.join(
    OUTPUT_DIR,
    "metadata.json"
)

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("\nDONE generating queries.")