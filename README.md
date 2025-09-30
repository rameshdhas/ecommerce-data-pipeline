# E-commerce Data Pipeline

An AWS CDK project that creates a serverless data pipeline for processing CSV files uploaded to S3, generating vector embeddings using AWS Bedrock, and storing them in a vector database.

## Architecture

1. **S3 Upload**: CSV files are uploaded to an S3 bucket
2. **Lambda Trigger**: S3 events trigger a Lambda function
3. **Glue Job**: Lambda starts a Glue ETL job to process the CSV
4. **LLM Integration**: Glue job calls AWS Bedrock to generate vector embeddings
5. **Vector Storage**: Processed data with embeddings is stored in a vector database
6. **Scheduled Processing**: EventBridge scheduler runs the pipeline daily at 2 AM UTC

## Components

- **Data Bucket**: S3 bucket for CSV file uploads
- **Scripts Bucket**: S3 bucket for Glue job scripts
- **Lambda Functions**:
  - Real-time trigger for individual CSV uploads
  - Scheduled batch processor for daily runs
- **Glue Job**: Processes CSV data and generates embeddings
- **EventBridge Scheduler**: Triggers daily processing at 2 AM UTC
- **IAM Roles**: Proper permissions for all services

## Prerequisites

- AWS CLI configured with appropriate permissions
- Node.js and yarn installed
- AWS CDK CLI installed (`npm install -g aws-cdk`)

## Deployment

1. Install dependencies:
   ```bash
   yarn install
   ```

2. Build the project:
   ```bash
   yarn build
   ```

3. Bootstrap CDK (first time only):
   ```bash
   npx cdk bootstrap
   ```

4. Deploy the stack:
   ```bash
   ./deploy.sh
   ```

## Usage

### Real-time Processing
1. After deployment, upload the Glue script to the scripts bucket:
   ```bash
   aws s3 cp glue-jobs/data_processing_job.py s3://[scripts-bucket-name]/
   ```

2. Upload a CSV file to the data bucket to trigger immediate processing:
   ```bash
   aws s3 cp your-data.csv s3://[data-bucket-name]/
   ```

### Scheduled Batch Processing
The pipeline automatically runs daily at 2 AM UTC to process files from the previous day. To use scheduled processing:

1. Upload CSV files to the daily folder structure:
   ```bash
   aws s3 cp your-data.csv s3://[data-bucket-name]/daily-uploads/YYYY/MM/DD/
   ```

2. Files will be automatically processed during the next scheduled run

### Monitoring
- Monitor real-time and scheduled Glue job executions in the AWS Glue console
- Check CloudWatch Logs for Lambda function and Glue job logs
- Review EventBridge rule metrics for scheduled execution status

## Configuration

### CSV File Format
The Glue job expects CSV files with headers. Customize the `create_text_content()` function in the Glue script based on your specific CSV structure.

### Vector Database
The current implementation saves embeddings as JSON files to S3. Replace the `save_to_vector_store()` function with your preferred vector database:
- Amazon OpenSearch with vector search
- Pinecone
- Weaviate
- Chroma
- PostgreSQL with pgvector

### LLM Model
The pipeline uses Amazon Bedrock's Titan embedding model by default. You can modify the `generate_embeddings()` function to use other models or endpoints.

## Useful Commands

* `yarn build`       - Compile TypeScript to JavaScript
* `yarn watch`       - Watch for changes and compile
* `yarn test`        - Run Jest unit tests
* `npx cdk deploy`   - Deploy the stack to AWS
* `npx cdk diff`     - Compare deployed stack with current state
* `npx cdk synth`    - Emit the synthesized CloudFormation template
* `npx cdk destroy`  - Destroy the stack and all resources

## Cost Considerations

- S3 storage costs for data and scripts
- Lambda invocation costs (minimal for trigger function)
- Glue job runtime costs (pay per minute)
- Bedrock model invocation costs
- Data transfer costs

## Security

- All S3 buckets have appropriate access controls
- IAM roles follow least privilege principle
- Glue jobs run with dedicated service roles
- No hardcoded credentials in code
