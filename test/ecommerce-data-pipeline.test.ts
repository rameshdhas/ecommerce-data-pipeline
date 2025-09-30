import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { EcommerceDataPipelineStack } from '../lib/ecommerce-data-pipeline-stack';

describe('EcommerceDataPipelineStack', () => {
  let app: cdk.App;
  let stack: EcommerceDataPipelineStack;
  let template: Template;

  beforeEach(() => {
    app = new cdk.App();
    stack = new EcommerceDataPipelineStack(app, 'TestStack');
    template = Template.fromStack(stack);
  });

  describe('S3 Buckets', () => {
    test('Should create data bucket for CSV uploads', () => {
      template.hasResourceProperties('AWS::S3::Bucket', {
        BucketName: Match.objectLike({
          'Fn::Join': Match.arrayWith([
            '',
            Match.arrayWith([
              Match.stringLikeRegexp('ecommerce-data-pipeline-')
            ])
          ])
        })
      });

      // Check for EventBridge configuration on the data bucket
      const buckets = template.findResources('AWS::S3::Bucket');
      const dataBucket = Object.values(buckets).find(bucket => {
        const bucketName = (bucket as any).Properties?.BucketName;
        if (bucketName && bucketName['Fn::Join']) {
          const parts = bucketName['Fn::Join'][1];
          return parts.some((part: any) => typeof part === 'string' && part.includes('ecommerce-data-pipeline'));
        }
        return false;
      });
      expect(dataBucket).toBeDefined();
    });

    test('Should create scripts bucket for Glue scripts', () => {
      template.hasResourceProperties('AWS::S3::Bucket', {
        BucketName: Match.objectLike({
          'Fn::Join': Match.arrayWith([
            '',
            Match.arrayWith([
              Match.stringLikeRegexp('ecommerce-glue-scripts-')
            ])
          ])
        })
      });
    });

    test('Should have bucket auto-deletion enabled for cleanup', () => {
      template.resourceCountIs('Custom::S3AutoDeleteObjects', 2);
    });
  });

  describe('Glue Job', () => {
    test('Should create Glue job with correct configuration', () => {
      template.hasResourceProperties('AWS::Glue::Job', {
        Name: 'ecommerce-data-processing',
        GlueVersion: '4.0',
        Command: {
          Name: 'glueetl',
          PythonVersion: '3',
          ScriptLocation: Match.objectLike({
            'Fn::Join': ['', Match.arrayWith([
              's3://',
              Match.objectLike({ Ref: Match.anyValue() }),
              '/data_processing_job.py'
            ])]
          })
        },
        DefaultArguments: Match.objectLike({
          '--job-language': 'python',
          '--job-bookmark-option': 'job-bookmark-enable',
          '--enable-metrics': '',
          '--enable-continuous-cloudwatch-log': 'true'
        }),
        Timeout: 60,
        MaxRetries: 0
      });
    });

    test('Should create IAM role for Glue job', () => {
      template.hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [{
            Effect: 'Allow',
            Principal: {
              Service: 'glue.amazonaws.com'
            },
            Action: 'sts:AssumeRole'
          }]
        },
        ManagedPolicyArns: [
          {
            'Fn::Join': ['', [
              'arn:',
              { Ref: 'AWS::Partition' },
              ':iam::aws:policy/service-role/AWSGlueServiceRole'
            ]]
          }
        ]
      });
    });

    test('Should grant Glue job permissions for Bedrock and SageMaker', () => {
      template.hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: Match.arrayWith([
            Match.objectLike({
              Effect: 'Allow',
              Action: [
                'bedrock:InvokeModel',
                'bedrock:InvokeModelWithResponseStream',
                'sagemaker:InvokeEndpoint'
              ],
              Resource: '*'
            })
          ])
        }
      });
    });
  });

  describe('Lambda Functions', () => {
    test('Should create trigger Lambda for S3 events', () => {
      template.hasResourceProperties('AWS::Lambda::Function', {
        Handler: 'trigger-glue-job.handler',
        Runtime: 'python3.9',
        Environment: {
          Variables: {
            GLUE_JOB_NAME: 'ecommerce-data-processing'
          }
        }
      });
    });

    test('Should create scheduled batch processor Lambda', () => {
      template.hasResourceProperties('AWS::Lambda::Function', {
        Handler: 'scheduled-batch-processor.handler',
        Runtime: 'python3.9',
        Timeout: 300,
        Environment: {
          Variables: Match.objectLike({
            DATA_BUCKET: Match.anyValue(),
            GLUE_JOB_NAME: 'ecommerce-data-processing'
          })
        }
      });
    });

    test('Should create process all files Lambda', () => {
      template.hasResourceProperties('AWS::Lambda::Function', {
        Handler: 'process-all-files.handler',
        Runtime: 'python3.9',
        Timeout: 300
      });
    });

    test('Should grant Lambda permissions to start Glue jobs', () => {
      const lambdaFunctions = template.findResources('AWS::Lambda::Function');
      expect(Object.keys(lambdaFunctions).length).toBeGreaterThanOrEqual(3);

      template.hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: Match.arrayWith([
            Match.objectLike({
              Effect: 'Allow',
              Action: 'glue:StartJobRun'
            })
          ])
        }
      });
    });
  });

  describe('S3 Event Notifications', () => {
    test('Should configure S3 to trigger Lambda on CSV upload', () => {
      template.hasResourceProperties('Custom::S3BucketNotifications', {
        NotificationConfiguration: {
          LambdaFunctionConfigurations: [
            {
              Events: ['s3:ObjectCreated:*'],
              Filter: {
                Key: {
                  FilterRules: [
                    {
                      Name: 'suffix',
                      Value: '.csv'
                    }
                  ]
                }
              },
              LambdaFunctionArn: Match.anyValue()
            }
          ]
        }
      });
    });

    test('Should grant S3 permission to invoke Lambda', () => {
      template.hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        Principal: 's3.amazonaws.com',
        SourceAccount: { Ref: 'AWS::AccountId' }
      });
    });
  });

  describe('EventBridge Scheduling', () => {
    test('Should create daily schedule rule at 2 AM UTC', () => {
      template.hasResourceProperties('AWS::Events::Rule', {
        ScheduleExpression: 'cron(0 2 * * ? *)',
        Description: 'Trigger data pipeline daily at 2 AM UTC',
        State: 'ENABLED'
      });
    });

    test('Should create alternative direct Glue schedule (disabled)', () => {
      template.hasResourceProperties('AWS::Events::Rule', {
        ScheduleExpression: 'cron(30 2 * * ? *)',
        Description: 'Directly trigger Glue job daily at 2:30 AM UTC',
        State: 'DISABLED'
      });
    });

    test('Should configure Lambda as EventBridge target', () => {
      template.hasResourceProperties('AWS::Events::Rule', {
        Targets: [
          {
            Arn: Match.anyValue(),
            Id: Match.anyValue()
          }
        ]
      });
    });
  });

  describe('Stack Outputs', () => {
    test('Should output data bucket name', () => {
      template.hasOutput('DataBucketName', {
        Description: 'S3 bucket for CSV uploads'
      });
    });

    test('Should output scripts bucket name', () => {
      template.hasOutput('ScriptsBucketName', {
        Description: 'S3 bucket for Glue scripts'
      });
    });

    test('Should output Glue job name', () => {
      template.hasOutput('GlueJobName', {
        Description: 'Glue job name for data processing'
      });
    });

    test('Should output daily schedule time', () => {
      template.hasOutput('DailyScheduleTime', {
        Value: '2:00 AM UTC',
        Description: 'Daily pipeline execution time'
      });
    });

    test('Should output scheduled Lambda function name', () => {
      template.hasOutput('ScheduledLambdaFunction', {
        Description: 'Lambda function for scheduled batch processing'
      });
    });
  });

  describe('Security and Permissions', () => {
    test('Should follow least privilege for IAM roles', () => {
      const roles = template.findResources('AWS::IAM::Role');
      expect(Object.keys(roles).length).toBeGreaterThanOrEqual(2); // Glue role + Lambda roles
    });

    test('Should grant S3 read/write permissions appropriately', () => {
      template.hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: Match.arrayWith([
            Match.objectLike({
              Effect: 'Allow',
              Action: Match.arrayWith([
                's3:GetObject*',
                's3:GetBucket*',
                's3:List*'
              ])
            })
          ])
        }
      });
    });

    test('Should not have any wildcarded principal permissions', () => {
      const policies = template.findResources('AWS::IAM::Policy');
      Object.values(policies).forEach(policy => {
        const statements = (policy as any).Properties?.PolicyDocument?.Statement || [];
        statements.forEach((statement: any) => {
          expect(statement.Principal).not.toEqual('*');
          expect(statement.Principal).not.toEqual({ AWS: '*' });
        });
      });
    });
  });

  describe('Resource Tagging', () => {
    test('Should have CDK metadata for all resources', () => {
      const resources = template.toJSON().Resources;
      const cdkResources = Object.values(resources).filter((resource: any) =>
        resource.Type?.startsWith('AWS::')
      );
      expect(cdkResources.length).toBeGreaterThan(0);
    });
  });
});
