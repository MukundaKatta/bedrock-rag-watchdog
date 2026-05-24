"""Tests for SageMaker module — all offline (stub mode)."""

import pytest
from ..sagemaker import submit_finetune_job, write_baseline_manifest, read_baseline_manifest


def test_submit_finetune_stub_returns_completed():
    job = submit_finetune_job(
        base_model="BAAI/bge-small-en-v1.5",
        training_data_s3="s3://my-bucket/train/",
        output_s3="s3://my-bucket/output/",
        stub=True,
    )
    assert job.status == "Completed"
    assert job.job_name.startswith("rag-embed-ft-")
    assert "model.tar.gz" in job.model_artifact_uri


def test_submit_finetune_job_name_prefix():
    job = submit_finetune_job(
        base_model="BAAI/bge-small-en-v1.5",
        training_data_s3="s3://bucket/data/",
        output_s3="s3://bucket/out/",
        job_name_prefix="custom-prefix",
        stub=True,
    )
    assert job.job_name.startswith("custom-prefix-")


def test_write_baseline_manifest_stub():
    from ..sagemaker import FineTuneJob
    job = FineTuneJob(
        job_name="test-job",
        model_artifact_uri="s3://bucket/model.tar.gz",
        status="Completed",
        baseline_drift_mean=0.15,
        baseline_drift_std=0.04,
    )
    uri = write_baseline_manifest(job, "my-bucket", stub=True)
    assert uri.startswith("s3://")
    assert "my-bucket" in uri


def test_read_baseline_manifest_stub_returns_dict():
    manifest = read_baseline_manifest("my-bucket", stub=True)
    assert "baseline_drift_mean" in manifest
    assert "job_name" in manifest
    assert isinstance(manifest["baseline_drift_mean"], float)


def test_read_baseline_manifest_stub_has_expected_keys():
    manifest = read_baseline_manifest("bucket", stub=True)
    expected = {"job_name", "model_artifact_uri", "baseline_drift_mean", "baseline_drift_std", "written_at", "version"}
    assert expected.issubset(set(manifest.keys()))


def test_baseline_drift_values_reasonable():
    job = submit_finetune_job("m", "s3://a", "s3://b", stub=True)
    assert 0.0 <= job.baseline_drift_mean <= 1.0
    assert 0.0 <= job.baseline_drift_std <= 1.0
