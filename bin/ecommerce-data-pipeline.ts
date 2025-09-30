#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { EcommerceDataPipelineStack } from '../lib/ecommerce-data-pipeline-stack';

const app = new cdk.App();

new EcommerceDataPipelineStack(app, 'EcommerceDataPipelineStack', {
   env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});