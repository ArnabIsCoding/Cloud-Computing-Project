import os
import io
import logging
import zipfile
import boto3
import pandas as pd
from typing import Dict, Any

# Configure structured logging
logger = logging.getLogger(__name__)
if os.environ.get('DEBUG') == 'true':
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Updated fields based on the Kaggle Data Expo 2009 dataset
SELECTED_FIELDS = [
    'Year', 'Month', 'DayofMonth', 'DayOfWeek',
    'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime',
    'UniqueCarrier', 'FlightNum'
]

class S3FlightDataCleaner:
    def __init__(self, src_bucket: str, dest_bucket: str, dest_prefix: str):
        self.src_bucket = src_bucket
        self.dest_bucket = dest_bucket
        self.dest_prefix = dest_prefix
        self.s3_client = boto3.client('s3')

    def download_and_extract_df(self, key: str) -> pd.DataFrame:
        """Downloads a zipfile from S3, extracts the CSV, and loads it into a Pandas DataFrame."""
        logger.info(f"Downloading object {self.src_bucket}/{key} from S3.")

        response = self.s3_client.get_object(Bucket=self.src_bucket, Key=key)
        zip_bytes = io.BytesIO(response['Body'].read())

        dataframes = []
        with zipfile.ZipFile(zip_bytes, mode='r') as z:
            for filename in z.namelist():
                if filename.endswith('.csv'):
                    logger.info(f"Loading CSV file {filename} into memory.")
                    # Read directly from the zip file into Pandas
                    with z.open(filename) as f:
                        # Use usecols to only load the data we actually need into memory
                        df = pd.read_csv(f, usecols=SELECTED_FIELDS, low_memory=False)
                        dataframes.append(df)

        if not dataframes:
            raise ValueError(f"No CSV files found in zip archive: {key}")

        return pd.concat(dataframes, ignore_index=True)

    def write_partitions_to_s3(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Groups data by Date and writes partitioned Parquet/CSV files back to S3."""
        results = {
            'written_lines': 0,
            'written_files': []
        }

        # Pandas handles the grouping natively, replacing the huge nested dictionaries
        grouped = df.groupby(['Year', 'Month', 'DayofMonth'])

        for (year, month, day), group_df in grouped:
            # Modern data pipelines prefer Parquet, but we can stick to CSV if required
            # Format: prefix/YYYY/MM/YYYY_MM_DD.csv
            output_key = f"{self.dest_prefix}/{year:04d}/{month:02d}/{year:04d}_{month:02d}_{day:02d}.csv"

            # Write dataframe to a string buffer
            csv_buffer = io.StringIO()
            group_df.to_csv(csv_buffer, index=False)

            logger.debug(f"Putting CSV file {output_key} onto S3.")
            self.s3_client.put_object(
                Bucket=self.dest_bucket,
                Key=output_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )

            results['written_lines'] += len(group_df)
            results['written_files'].append({
                'bucketname': self.dest_bucket,
                'key': output_key
            })

        logger.info(f"Successfully partitioned and wrote {results['written_lines']} lines to S3.")
        return results

    def execute(self, key: str) -> bool:
        """Executes the full extraction, transformation, and load (ETL) process."""
        try:
            df = self.download_and_extract_df(key)

            # Additional cleaning logic can go here (e.g., dropping NaNs)
            initial_rows = len(df)
            df.dropna(subset=['Year', 'Month', 'DayofMonth'], inplace=True)

            self.write_partitions_to_s3(df)

            logger.info(f"Processing completed: Read {initial_rows} lines.")
            return True
        except Exception as e:
            logger.error(f"Error processing {key}: {str(e)}")
            return False


def handle_zipfile(event: dict, context: Any) -> dict:
    """
    AWS Lambda entry point.
    event is expected as dictionary:
    {
        'src-bucketname': <s3-bucket>,
        'key': <location-of-zipfile>,
        'dst-bucketname': <s3-bucket>,
        'dst-key-prefix': <prefix>
    }
    """
    logger.info(f"Lambda Handler invoked for {event.get('src-bucketname')}/{event.get('key')}")

    cleaner = S3FlightDataCleaner(
        src_bucket=event['src-bucketname'],
        dest_bucket=event['dst-bucketname'],
        dest_prefix=event['dst-key-prefix']
    )

    success = cleaner.execute(key=event['key'])

    return {
        'status': 'OK' if success else 'ERROR'
    }
