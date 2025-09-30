import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as glue from 'aws-cdk-lib/aws-glue';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3n from 'aws-cdk-lib/aws-s3-notifications';
import { Construct } from 'constructs';

export class EcommerceDataPipelineStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // S3 bucket for CSV uploads
    const dataBucket = new s3.Bucket(this, 'DataBucket', {
      bucketName: `ecommerce-data-pipeline-${this.account}-${this.region}`,
      eventBridgeEnabled: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // S3 bucket for Glue scripts
    const scriptsBucket = new s3.Bucket(this, 'ScriptsBucket', {
      bucketName: `ecommerce-glue-scripts-${this.account}-${this.region}`,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // IAM role for Glue job
    const glueRole = new iam.Role(this, 'GlueJobRole', {
      assumedBy: new iam.ServicePrincipal('glue.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSGlueServiceRole'),
      ],
    });

    // Grant Glue access to S3 buckets
    dataBucket.grantReadWrite(glueRole);
    scriptsBucket.grantRead(glueRole);

    // Add permissions for LLM endpoint (Bedrock/SageMaker)
    glueRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:InvokeModel',
        'bedrock:InvokeModelWithResponseStream',
        'sagemaker:InvokeEndpoint',
      ],
      resources: ['*'],
    }));

    // Glue job for processing CSV and generating vectors
    const glueJob = new glue.CfnJob(this, 'DataProcessingJob', {
      name: 'ecommerce-data-processing',
      role: glueRole.roleArn,
      command: {
        name: 'glueetl',
        scriptLocation: `s3://${scriptsBucket.bucketName}/data_processing_job.py`,
        pythonVersion: '3',
      },
      defaultArguments: {
        '--job-language': 'python',
        '--job-bookmark-option': 'job-bookmark-enable',
        '--enable-metrics': '',
        '--enable-continuous-cloudwatch-log': 'true',
        '--data-bucket': dataBucket.bucketName,
      },
      maxRetries: 0,
      timeout: 60,
      glueVersion: '4.0',
    });

    // Lambda function to trigger Glue job on S3 upload
    const triggerFunction = new lambda.Function(this, 'TriggerGlueJob', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'index.handler',
      code: lambda.Code.fromInline(`
import json
import boto3

glue_client = boto3.client('glue')

def handler(event, context):
    # Parse S3 event
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        # Only process CSV files
        if key.endswith('.csv'):
            try:
                # Start Glue job
                response = glue_client.start_job_run(
                    JobName='${glueJob.name}',
                    Arguments={
                        '--input-path': f's3://{bucket}/{key}'
                    }
                )
                print(f"Started Glue job {response['JobRunId']} for {key}")
            except Exception as e:
                print(f"Error starting Glue job: {str(e)}")
                raise

    return {
        'statusCode': 200,
        'body': json.dumps('Successfully triggered Glue job')
    }
      `),
      environment: {
        'GLUE_JOB_NAME': glueJob.name || 'ecommerce-data-processing',
      },
    });

    // Grant Lambda permission to start Glue jobs
    triggerFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['glue:StartJobRun'],
      resources: [
        `arn:aws:glue:${this.region}:${this.account}:job/${glueJob.name}`,
      ],
    }));

    // S3 event notification to trigger Lambda
    dataBucket.addEventNotification(
      s3.EventType.OBJECT_CREATED,
      new s3n.LambdaDestination(triggerFunction),
      { suffix: '.csv' }
    );

    // Outputs
    new cdk.CfnOutput(this, 'DataBucketName', {
      value: dataBucket.bucketName,
      description: 'S3 bucket for CSV uploads',
    });

    new cdk.CfnOutput(this, 'ScriptsBucketName', {
      value: scriptsBucket.bucketName,
      description: 'S3 bucket for Glue scripts',
    });

    new cdk.CfnOutput(this, 'GlueJobName', {
      value: glueJob.name || 'ecommerce-data-processing',
      description: 'Glue job name for data processing',
    });
  }
}
