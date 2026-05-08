import io
import zipfile
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.batch.clean_data import S3FlightDataCleaner, handle_zipfile

# ==========================================
# Fixtures (Reusable Test Data)
# ==========================================

@pytest.fixture
def dummy_flight_data():
    """Creates a small Pandas DataFrame representing raw flight data."""
    return pd.DataFrame({
        'Year': [2008, 2008, 2008],
        'Month': [1, 1, 2],
        'DayofMonth': [15, 16, 1],
        'DayOfWeek': [2, 3, 5],
        'DepTime': [1430, None, 1000],
        'CRSDepTime': [1430, 1500, 1000],
        'ArrTime': [1600, None, 1130],
        'CRSArrTime': [1600, 1630, 1130],
        'UniqueCarrier': ['WN', 'AA', 'DL'],
        'FlightNum': [101, 202, 303]
    })

@pytest.fixture
def mock_zip_bytes():
    """Creates an in-memory zip file containing a dummy CSV."""
    csv_content = (
        "Year,Month,DayofMonth,DayOfWeek,DepTime,CRSDepTime,ArrTime,CRSArrTime,UniqueCarrier,FlightNum\n"
        "2008,1,15,2,1430,1430,1600,1600,WN,101\n"
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('2008.csv', csv_content)

    zip_buffer.seek(0)
    return zip_buffer.read()

# ==========================================
# Unit Tests
# ==========================================

class TestS3FlightDataCleaner:

    @patch('src.batch.clean_data.boto3.client')
    def test_initialization(self, mock_boto_client):
        """Test that the cleaner initializes with correct parameters."""
        cleaner = S3FlightDataCleaner('source-bucket', 'dest-bucket', 'processed')

        assert cleaner.src_bucket == 'source-bucket'
        assert cleaner.dest_bucket == 'dest-bucket'
        assert cleaner.dest_prefix == 'processed'
        mock_boto_client.assert_called_once_with('s3')

    @patch('src.batch.clean_data.boto3.client')
    def test_download_and_extract_df(self, mock_boto_client, mock_zip_bytes):
        """Test that the cleaner correctly downloads and parses a CSV inside a ZIP from S3."""
        # Setup the mock S3 response
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            'Body': MagicMock(read=MagicMock(return_value=mock_zip_bytes))
        }
        mock_boto_client.return_value = mock_s3

        cleaner = S3FlightDataCleaner('src', 'dest', 'prefix')

        # Execute
        df = cleaner.download_and_extract_df('raw/flights.zip')

        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]['UniqueCarrier'] == 'WN'
        mock_s3.get_object.assert_called_once_with(Bucket='src', Key='raw/flights.zip')

    @patch('src.batch.clean_data.boto3.client')
    def test_write_partitions_to_s3(self, mock_boto_client, dummy_flight_data):
        """Test that data is grouped correctly by date and written to S3."""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        cleaner = S3FlightDataCleaner('src', 'dest', 'prefix')

        # Execute
        results = cleaner.write_partitions_to_s3(dummy_flight_data)

        # Assertions
        # The dummy data has 3 rows across 3 distinct dates, so 3 files should be written
        assert results['written_lines'] == 3
        assert len(results['written_files']) == 3
        assert mock_s3.put_object.call_count == 3

        # Verify the S3 key format for the first partition (2008-01-15)
        expected_key = "prefix/2008/01/2008_01_15.csv"
        assert results['written_files'][0]['key'] == expected_key

    @patch.object(S3FlightDataCleaner, 'write_partitions_to_s3')
    @patch.object(S3FlightDataCleaner, 'download_and_extract_df')
    def test_execute_success(self, mock_download, mock_write, dummy_flight_data):
        """Test the full orchestration flow of the cleaner class."""
        mock_download.return_value = dummy_flight_data

        cleaner = S3FlightDataCleaner('src', 'dest', 'prefix')

        # Execute
        success = cleaner.execute('dummy_key.zip')

        # Assertions
        assert success is True
        mock_download.assert_called_once_with('dummy_key.zip')
        mock_write.assert_called_once()

    @patch.object(S3FlightDataCleaner, 'download_and_extract_df')
    def test_execute_failure(self, mock_download):
        """Test that the cleaner handles errors gracefully."""
        mock_download.side_effect = Exception("S3 Connection Error")

        cleaner = S3FlightDataCleaner('src', 'dest', 'prefix')

        # Execute
        success = cleaner.execute('dummy_key.zip')

        # Assertions
        assert success is False


# ==========================================
# Lambda Handler Tests
# ==========================================

@patch('src.batch.clean_data.S3FlightDataCleaner')
def test_handle_zipfile_lambda(mock_cleaner_class):
    """Test the AWS Lambda entry point."""
    # Setup mock cleaner instance
    mock_instance = MagicMock()
    mock_instance.execute.return_value = True
    mock_cleaner_class.return_value = mock_instance

    event = {
        'src-bucketname': 'input-bucket',
        'key': 'data.zip',
        'dst-bucketname': 'output-bucket',
        'dst-key-prefix': 'out/'
    }

    # Execute
    response = handle_zipfile(event, None)

    # Assertions
    assert response['status'] == 'OK'
    mock_cleaner_class.assert_called_once_with(
        src_bucket='input-bucket',
        dest_bucket='output-bucket',
        dest_prefix='out/'
    )
    mock_instance.execute.assert_called_once_with(key='data.zip')
