import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as glue from 'aws-cdk-lib/aws-glue';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3n from 'aws-cdk-lib/aws-s3-notifications';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as path from 'path';
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
        '--data_bucket': dataBucket.bucketName,
        '--elasticsearch_endpoint': 'https://ecommerce-project-a814cd.es.us-east-1.aws.elastic.cloud:443',
        '--elasticsearch_api_key': 'alBSNG41a0JUamwwcHZqNnoxaUs6Q2I0Ym5iNE14TkRvNEtDUXcyZF83Zw==',
        // Amazon Bedrock Foundation Model options (no external API keys needed):
        // Text Embeddings:
        // - 'amazon.titan-embed-text-v2': 8192 tokens, 1024 dimensions (configurable 256/512/1024)
        // - 'amazon.titan-embed-text-v1': 8192 tokens, 1536 dimensions
        // - 'cohere.embed-english-v3': 512 tokens, 1024 dimensions
        // - 'cohere.embed-multilingual-v3': 512 tokens, 1024 dimensions
        // Multimodal Embeddings:
        // - 'amazon.titan-embed-image-v1': Text + Image, 1024 dimensions
        // Text Summarization (for long content):
        // - 'anthropic.claude-3-haiku-20240307': 200K tokens (summarizes then uses Titan for embeddings)
        // - 'anthropic.claude-3-sonnet-20240229': 200K tokens (summarizes then uses Titan for embeddings)
        '--embedding_model': 'amazon.titan-embed-text-v1',
        '--additional-python-modules': 'boto3>=1.34.0,requests,numpy',
      },
      maxRetries: 0,
      timeout: 60,
      glueVersion: '4.0',
    });

    // Lambda function to trigger Glue job on S3 upload
    const triggerFunction = new lambda.Function(this, 'TriggerGlueJob', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'trigger-glue-job.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, 'functions')),
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

    // Lambda function to process scheduled batch jobs
    const scheduledProcessFunction = new lambda.Function(this, 'ScheduledBatchProcessor', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'scheduled-batch-processor.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, 'functions')),
      environment: {
        'DATA_BUCKET': dataBucket.bucketName,
        'GLUE_JOB_NAME': glueJob.name || 'ecommerce-data-processing',
      },
      timeout: cdk.Duration.minutes(5),
    });

    // Grant permissions to scheduled Lambda
    dataBucket.grantRead(scheduledProcessFunction);
    scheduledProcessFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['glue:StartJobRun'],
      resources: [
        `arn:aws:glue:${this.region}:${this.account}:job/${glueJob.name}`,
      ],
    }));

    // EventBridge rule to run daily at 2 AM UTC
    const dailyScheduleRule = new events.Rule(this, 'DailyPipelineSchedule', {
      schedule: events.Schedule.cron({
        minute: '0',
        hour: '2',
        day: '*',
        month: '*',
        year: '*'
      }),
      description: 'Trigger data pipeline daily at 2 AM UTC',
    });

    // Add Lambda as target for the EventBridge rule
    dailyScheduleRule.addTarget(new targets.LambdaFunction(scheduledProcessFunction));

    // Alternative: Direct Glue job scheduling (without Lambda)
    const directGlueScheduleRule = new events.Rule(this, 'DirectGlueJobSchedule', {
      schedule: events.Schedule.cron({
        minute: '30',
        hour: '2',
        day: '*',
        month: '*',
        year: '*'
      }),
      description: 'Directly trigger Glue job daily at 2:30 AM UTC',
      enabled: false, // Disabled by default, enable if you prefer direct Glue triggering
    });

    // Create a Lambda to process all files in the bucket (for direct Glue scheduling)
    const processAllFilesFunction = new lambda.Function(this, 'ProcessAllFiles', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'process-all-files.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, 'functions')),
      environment: {
        'DATA_BUCKET': dataBucket.bucketName,
        'GLUE_JOB_NAME': glueJob.name || 'ecommerce-data-processing',
      },
      timeout: cdk.Duration.minutes(5),
    });

    dataBucket.grantRead(processAllFilesFunction);
    processAllFilesFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['glue:StartJobRun'],
      resources: [
        `arn:aws:glue:${this.region}:${this.account}:job/${glueJob.name}`,
      ],
    }));

    directGlueScheduleRule.addTarget(new targets.LambdaFunction(processAllFilesFunction));

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

    new cdk.CfnOutput(this, 'DailyScheduleTime', {
      value: '2:00 AM UTC',
      description: 'Daily pipeline execution time',
    });

    new cdk.CfnOutput(this, 'ScheduledLambdaFunction', {
      value: scheduledProcessFunction.functionName,
      description: 'Lambda function for scheduled batch processing',
    });
  }
}
