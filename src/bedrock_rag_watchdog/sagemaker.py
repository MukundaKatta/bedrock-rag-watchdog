"""
SageMaker embedding fine-tune integration.

Used by the AWS AI League submission (Model Customization track).
Provides stubs for:
  - Submitting a fine-tune job for an embedding model
  - Polling job status
  - Writing the fine-tune manifest to S3 (for AgentCore drift baseline)

In stub mode (default) all jobs are simulated and no AWS calls are made.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FineTuneJob:
    job_name: str
    model_artifact_uri: str
    status: str  # "InProgress" | "Completed" | "Failed"
    baseline_drift_mean: float = 0.0
    baseline_drift_std: float = 0.0
    completed_at: Optional[float] = None


def submit_finetune_job(
    base_model: str,
    training_data_s3: str,
    output_s3: str,
    job_name_prefix: str = "rag-embed-ft",
    stub: bool = True,
) -> FineTuneJob:
    """
    Submit a SageMaker fine-tune job for an embedding model.

    In stub mode: returns a fake completed job instantly.
    In production: calls sagemaker.create_training_job().
    """
    job_name = f"{job_name_prefix}-{int(time.time())}"

    if stub:
        print(f"[stub] SageMaker fine-tune job submitted: {job_name}")
        print(f"[stub]   base_model={base_model}")
        print(f"[stub]   training_data={training_data_s3}")
        return FineTuneJob(
            job_name=job_name,
            model_artifact_uri=f"{output_s3}/{job_name}/output/model.tar.gz",
            status="Completed",
            baseline_drift_mean=0.12,
            baseline_drift_std=0.03,
            completed_at=time.time(),
        )

    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("boto3 not installed") from e

    sm = boto3.client("sagemaker")
    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04",
            "TrainingInputMode": "File",
        },
        RoleArn="arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole",
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": training_data_s3}},
            }
        ],
        OutputDataConfig={"S3OutputPath": output_s3},
        ResourceConfig={
            "InstanceType": "ml.p3.2xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        HyperParameters={"base_model_name": base_model},
    )
    return FineTuneJob(
        job_name=job_name,
        model_artifact_uri=f"{output_s3}/{job_name}/output/model.tar.gz",
        status="InProgress",
    )


def write_baseline_manifest(
    job: FineTuneJob,
    s3_bucket: str,
    s3_key: str = "rag-drift-baseline/manifest.json",
    stub: bool = True,
) -> str:
    """
    Write the fine-tune baseline manifest to S3.

    The AgentCore drift agent reads this on startup to know what the
    pre-fine-tune embedding distribution looked like.
    """
    manifest = {
        "job_name": job.job_name,
        "model_artifact_uri": job.model_artifact_uri,
        "baseline_drift_mean": job.baseline_drift_mean,
        "baseline_drift_std": job.baseline_drift_std,
        "written_at": time.time(),
        "version": "1",
    }
    s3_uri = f"s3://{s3_bucket}/{s3_key}"

    if stub:
        print(f"[stub] Baseline manifest written to {s3_uri}")
        print(f"[stub]   {json.dumps(manifest, indent=2)}")
        return s3_uri

    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("boto3 not installed") from e

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=s3_bucket,
        Key=s3_key,
        Body=json.dumps(manifest).encode(),
        ContentType="application/json",
    )
    return s3_uri


def read_baseline_manifest(
    s3_bucket: str,
    s3_key: str = "rag-drift-baseline/manifest.json",
    stub: bool = True,
) -> Dict[str, Any]:
    """
    Read the baseline manifest written after the last fine-tune.
    Used by the AgentCore agent to set its drift comparison baseline.
    """
    if stub:
        return {
            "job_name": "rag-embed-ft-stub",
            "model_artifact_uri": "s3://stub-bucket/model.tar.gz",
            "baseline_drift_mean": 0.12,
            "baseline_drift_std": 0.03,
            "written_at": time.time() - 86400,
            "version": "1",
        }

    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("boto3 not installed") from e

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    return json.loads(obj["Body"].read())
