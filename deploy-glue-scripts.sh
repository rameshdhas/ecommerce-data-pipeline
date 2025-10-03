#!/bin/bash

# Script to deploy Glue job scripts and dependencies to S3

# Get the scripts bucket name from CDK output
SCRIPTS_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name EcommerceDataPipelineStack \
  --query 'Stacks[0].Outputs[?OutputKey==`ScriptsBucketName`].OutputValue' \
  --output text)

if [ -z "$SCRIPTS_BUCKET" ]; then
  echo "Error: Could not find ScriptsBucketName from CloudFormation stack"
  echo "Make sure the stack is deployed first with: cdk deploy"
  exit 1
fi

echo "Using scripts bucket: $SCRIPTS_BUCKET"

# Create a temporary directory for packaging
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Copy all Python modules to temp directory
cp -r glue-jobs/* "$TEMP_DIR/"

# Create the ZIP file containing all modules
cd "$TEMP_DIR"
zip -r glue-jobs.zip . -x "__pycache__/*" "*.pyc"

# Upload the main script
aws s3 cp data_processing_job.py "s3://$SCRIPTS_BUCKET/data_processing_job.py"

# Upload the ZIP file with all modules
aws s3 cp glue-jobs.zip "s3://$SCRIPTS_BUCKET/glue-jobs.zip"

# Clean up
cd - > /dev/null
rm -rf "$TEMP_DIR"

echo "Successfully deployed Glue scripts to S3 bucket: $SCRIPTS_BUCKET"
echo "Main script: s3://$SCRIPTS_BUCKET/data_processing_job.py"
echo "Dependencies: s3://$SCRIPTS_BUCKET/glue-jobs.zip"