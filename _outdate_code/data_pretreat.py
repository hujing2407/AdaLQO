
import sys
import AdaLQO.utils as utils


sys.path.insert(0, '../bao_server')


def main():
    df = utils.load_data("tpc-ds")
    print(f"All data loaded: {df.shape}")

    # Filter queries that use PostgreSQL's default execution timeout
    df_psql_overtime = df[df.apply(lambda x: x['latency_list'][0] is None, axis=1)]
    print(f"PSQL Timeout queries: {df_psql_overtime.shape}")

    df = df[df.apply(lambda x: x['latency_list'][0] is not None, axis=1)]
    # print(df.shape)
    # Remove inf latencies
    df["latency_list"] = df["latency_list"].apply(
        lambda xs: [x for x in xs if x is not None] if isinstance(xs, list) else []
    )

    mismatched_rows = df[df.apply(lambda x: len(x['latency_list']) != len(x['plans']), axis=1)]
    # varify all latency/plan paired
    assert(mismatched_rows.shape[0] == 0)

    # Remove empty row (all plans timeout)
    df = df[
        df["latency_list"].apply(lambda xs: isinstance(xs, list) and len(xs) > 0)
        & df["plans"].apply(lambda xs: isinstance(xs, list) and len(xs) > 0)
    ]
    df = df.reset_index(drop=True)

    print(df.shape)
    df.to_pickle("dataset/tpc-ds/data_df.pkl")

    print("Queries execution data have been saved into dataset/tpc-ds/data_df.pkl")

if __name__ == "__main__":
    main()
