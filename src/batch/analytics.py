import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, round, sum, rank, concat_ws
from pyspark.sql.window import Window

# Configure structured logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FlightAnalytics:
    def __init__(self, app_name: str = "FlightDataAnalytics"):
        # Modern Spark handles parallelization automatically without the need
        # for manual 'SET hive.exec.parallel=true' commands
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")

    def load_data(self, path: str):
        """Loads the cleaned dataset into a Spark DataFrame."""
        logger.info(f"Loading data from {path}")
        return self.spark.read.parquet(path)

    def run_group_1_queries(self, df):
        """Runs the Group 1 aggregate analytics queries."""
        logger.info("Executing Group 1 Analytics...")

        # We only look at non-cancelled flights for these queries
        if "Cancelled" in df.columns:
            active_flights = df.filter(col("Cancelled") == 0)
        else:
            active_flights = df

        # Question 1.1: Busiest Airports (Departures + Arrivals)
        departures = active_flights.groupBy("Origin").agg(count("FlightNum").alias("total_departing"))
        arrivals = active_flights.groupBy("Dest").agg(count("FlightNum").alias("total_arriving"))

        q1_1 = departures.join(arrivals, departures.Origin == arrivals.Dest) \
            .select(
                col("Origin").alias("airport"),
                (col("total_departing") + col("total_arriving")).alias("total_flights")
            ).orderBy(col("total_flights").desc())

        logger.info("\n--- Q1.1 Top 5 Busiest Airports ---")
        q1_1.show(5)

        # Question 1.2: Average Delay by Airline
        q1_2 = active_flights.groupBy("UniqueCarrier") \
            .agg(round(avg("ArrDelay"), 2).alias("avg_delay")) \
            .orderBy(col("avg_delay").asc())

        logger.info("\n--- Q1.2 Best Airlines by Average Arrival Delay ---")
        q1_2.show(5)

        # Question 1.3: Average Delay by Day of Week
        q1_3 = active_flights.groupBy("DayOfWeek") \
            .agg(round(avg("ArrDelay"), 2).alias("avg_delay")) \
            .orderBy(col("avg_delay").asc())

        logger.info("\n--- Q1.3 Best Days of the Week to Fly ---")
        q1_3.show(7)

        return q1_1, q1_2, q1_3

    def run_group_2_queries(self, df):
        """Runs the Group 2 ranking analytics queries (Window Functions)."""
        logger.info("Executing Group 2 Analytics...")

        if "Cancelled" in df.columns:
            active_flights = df.filter((col("Cancelled") == 0) & (col("DepDelay").isNotNull()))
        else:
            active_flights = df.filter(col("DepDelay").isNotNull())

        # Question 2.1: Top 10 Airlines by Airport (Lowest Departure Delay)
        window_spec_2_1 = Window.partitionBy("Origin").orderBy("avg_delay", "UniqueCarrier")

        q2_1 = active_flights.groupBy("Origin", "UniqueCarrier") \
            .agg(round(avg("DepDelay"), 2).alias("avg_delay")) \
            .withColumn("rank", rank().over(window_spec_2_1)) \
            .filter(col("rank") <= 10)

        logger.info("\n--- Q2.1 Top Airlines at SFO (Example) ---")
        q2_1.filter(col("Origin") == "SFO").show(5)

        # Question 2.2: Top 10 Destinations from each Airport (Lowest Delay)
        window_spec_2_2 = Window.partitionBy("Origin").orderBy("avg_delay", "Dest")

        q2_2 = active_flights.groupBy("Origin", "Dest") \
            .agg(round(avg("DepDelay"), 2).alias("avg_delay")) \
            .withColumn("rank", rank().over(window_spec_2_2)) \
            .filter(col("rank") <= 10)

        logger.info("\n--- Q2.2 Best Destinations from SFO (Example) ---")
        q2_2.filter(col("Origin") == "SFO").show(5)

        return q2_1, q2_2

    def save_results(self, df, output_path: str):
        """Saves analytical tables to Parquet."""
        logger.info(f"Saving results to {output_path}")
        df.write.mode("overwrite").parquet(output_path)

    def execute(self, input_path: str, output_dir: str):
        """Orchestrates the analytics pipeline."""
        df = self.load_data(input_path)

        # Cache the dataframe since we hit it multiple times
        df.cache()

        q1_1, q1_2, q1_3 = self.run_group_1_queries(df)
        q2_1, q2_2 = self.run_group_2_queries(df)

        os.makedirs(output_dir, exist_ok=True)
        self.save_results(q1_1, f"{output_dir}/q1_1_busiest_airports.parquet")
        self.save_results(q2_1, f"{output_dir}/q2_1_top_airlines_by_airport.parquet")

        # --- NEW: Extract features for the ML Pipeline ---
        logger.info("Generating Machine Learning Feature Set...")
        ml_features = df.select("Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepDelay", "ArrDelay").na.drop()
        self.save_results(ml_features, f"{output_dir}/ml_features.parquet")

        logger.info("Analytics pipeline completed successfully.")
        self.spark.stop()

if __name__ == "__main__":
    INPUT_FILE = "data/processed/cleaned_flights.parquet"
    OUTPUT_DIRECTORY = "data/analytics"

    analytics_job = FlightAnalytics()

    # Actually executing it this time!
    analytics_job.execute(INPUT_FILE, OUTPUT_DIRECTORY)
