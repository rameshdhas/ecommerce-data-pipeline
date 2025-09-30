#!/bin/bash

set -e

echo "🚀 Deploying E-commerce Data Pipeline..."

# Build the project
echo "📦 Building CDK project..."
yarn build

# Upload Glue script to S3 (will be created after first deploy)
echo "📝 Preparing Glue job script..."

# Synthesize CloudFormation template
echo "🔧 Synthesizing CDK stack..."
npx cdk synth

# Deploy the stack
echo "🚢 Deploying infrastructure..."
npx cdk deploy --require-approval never

echo "✅ Deployment complete!"

# Get stack outputs
echo "📊 Stack outputs:"
aws cloudformation describe-stacks \
  --stack-name EcommerceDataPipelineStack \
  --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
  --output table || echo "Stack outputs will be available after first deployment"

echo ""
echo "📋 Next steps:"
echo "1. Upload the Glue script to the scripts bucket"
echo "2. Upload a CSV file to the data bucket to trigger the pipeline"
echo "3. Monitor the Glue job execution in the AWS console"