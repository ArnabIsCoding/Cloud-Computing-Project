import os
import json
import logging
import boto3
from typing import Any, Dict

# Configure structured logging
logger = logging.getLogger(__name__)
if os.environ.get('DEBUG') == 'true':
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')

# It's best practice to pass the target Lambda name via environment variables
# rather than hardcoding it, but we provide a default fallback here.
TARGET_LAMBDA = os.environ.get('TARGET_LAMBDA_NAME', 'capstone-cleaning-dev-handle_zipfile')

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda Orchestrator.
    Scans an S3 bucket for zip files and asynchronously triggers a processing Lambda for each.
    """
    logger.info('Invoking get_zipfiles orchestrator handler.')

    src_bucket = event.get('src-bucketname')
    src_prefix = event.get('src-prefix', '')
    dst_bucket = event.get('dst-bucketname')
    dst_prefix = event.get('dst-prefix', '')

    if not src_bucket or not dst_bucket:
        logger.error("Missing required bucket names in event payload.")
        return {'statusCode': 400, 'body': 'Missing bucket parameters'}

    # Use a Paginator to handle buckets with more than 1,000 objects
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=src_bucket, Prefix=src_prefix)

    nr_success = 0
    nr_failed = 0

    try:
        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']

                # Only process .zip files
                if key.endswith('.zip'):
                    logger.info(f'Found ZIP file: {key}')

                    payload = {
                        'src-bucketname': src_bucket,
                        'key': key,
                        'dst-bucketname': dst_bucket,
                        'dst-key-prefix': dst_prefix
                    }

                    try:
                        # Modern AWS asynchronous invocation method
                        response = lambda_client.invoke(
                            FunctionName=TARGET_LAMBDA,
                            InvocationType='Event',  # 'Event' means asynchronous execution
                            Payload=json.dumps(payload)
                        )

                        # 202 is the HTTP status code for "Accepted" (Async trigger successful)
                        if response['StatusCode'] == 202:
                            logger.info(f'Successfully triggered processing Lambda for {key}')
                            nr_success += 1
                        else:
                            logger.error(f'Unexpected status code {response["StatusCode"]} for {key}')
                            nr_failed += 1

                    except Exception as e:
                        logger.error(f'Failed to invoke lambda for {key}: {str(e)}')
                        nr_failed += 1

    except Exception as e:
        logger.error(f"Failed to list objects in bucket {src_bucket}: {str(e)}")
        return {'statusCode': 500, 'body': str(e)}

    logger.info(f'Orchestration complete. Lambdas Started: {nr_success}, Failed to Start: {nr_failed}')

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Orchestration finished',
            'lambdas_started': nr_success,
            'lambdas_failed': nr_failed
        })
    }
