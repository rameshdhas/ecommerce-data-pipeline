from pyspark.sql import DataFrame
from awsglue.context import GlueContext


def process_csv_data(input_path: str, glue_context: GlueContext) -> DataFrame:
    """
    Read and process CSV data from S3
    """
    print(f"Processing CSV file: {input_path}")

    # Read CSV file using Glue DataCatalog or direct S3 read
    try:
        # Create dynamic frame from S3
        dynamic_frame = glue_context.create_dynamic_frame.from_options(
            connection_type="s3",
            connection_options={
                "paths": [input_path],
                "recurse": True
            },
            format="csv",
            format_options={
                "withHeader": True,
                "separator": ","
            }
        )

        # Convert to Spark DataFrame
        df = dynamic_frame.toDF()
        print(f"Successfully read {df.count()} rows from CSV")
        return df

    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        raise