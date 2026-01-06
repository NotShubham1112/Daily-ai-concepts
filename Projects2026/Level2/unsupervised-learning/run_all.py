from kmeans.kmeans_sklearn_compare import compare as kmeans_compare
from dbscan.dbscan_vs_kmeans import compare as dbscan_compare
from dimensionality_reduction.comparison import reduce_and_plot
from data.load_data import load_synthetic

def main():
    X, y = load_synthetic()

    print("Running K-Means comparison...")
    kmeans_compare()

    print("Running DBSCAN vs K-Means...")
    dbscan_compare(X)

    print("Running Dimensionality Reduction Comparison...")
    reduce_and_plot(X, y)

if __name__ == "__main__":
    main()
