import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, count, avg
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

def main():
    # 1. Initialize Spark Session with Kafka package
    spark = SparkSession.builder \
        .appName("ModernFlightStreamingAnalytics") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    KAFKA_BROKER = os.getenv("KAFKA_HOSTS", "localhost:9092")
    KAFKA_TOPIC = "flight_telemetry"

    # 2. Define schema based on the Kaggle dataset fields you need
    schema = StructType([
        StructField("Year", StringType(), True),
        StructField("Month", StringType(), True),
        StructField("DayofMonth", StringType(), True),
        StructField("DayOfWeek", StringType(), True),
        StructField("Origin", StringType(), True),
        StructField("Dest", StringType(), True),
        StructField("UniqueCarrier", StringType(), True),
        StructField("ArrDelay", DoubleType(), True) # Using Double for averages
    ])

    # 3. Read stream from Kafka
    raw_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "earliest") \
        .load()

    # 4. Parse the JSON payload
    parsed_df = raw_df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    # Clean the data: Replace null ArrDelay with 0.0
    clean_df = parsed_df.fillna({"ArrDelay": 0.0})

    # --- Solution 1.2: Average Delay by Carrier ---
    carrier_delay = clean_df.groupBy("UniqueCarrier") \
        .agg(
            avg("ArrDelay").alias("AvgDelay"),
            count("UniqueCarrier").alias("NrFlights")
        )

    # --- Solution 1.3: Average Delay by Day of Week ---
    day_delay = clean_df.groupBy("DayOfWeek") \
        .agg(
            avg("ArrDelay").alias("AvgDelay"),
            count("DayOfWeek").alias("NrFlights")
        )

    # 5. Write the output streams to the console
    query_carrier = carrier_delay.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    query_day = day_delay.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    # Keep the application running to process the stream
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()
