# Environment setup

Ensure micromamba is installed. Set up the environment by running

```bash
micromamba create --file environment.yml --yes
```

Note: be sure to change the `upload_dir` in `policy_inference_after_training.py` to correspond with the cluster being used (i.e., `s3://` for AWS and `gs://` for GCP).

# AWS Cluster

Create the cluster with

```bash
ray up cluster_aws.yml --yes --verbose
```

Allow use of `ray.init("auto")` and get access to the dashboard with

```bash
ray dashboard cluster_aws.yml
```

Submit jobs to the cluster with

```bash
ray job submit --runtime-env-json '{"working_dir":".","conda":"environment.yml","env_vars":{"AWS_ACCESS_KEY_ID":"'$(aws configure get aws_access_key_id)'","AWS_SECRET_ACCESS_KEY":"'$(aws configure get aws_secret_access_key)'"}}' --no-wait --verbose -- python policy_inference_after_training.py
```

*Note:* Uses the default credentials in `~/.aws`. These credentials are largely used to get saving to S3 working. Requires the `aws` CLI to be installed. (This has only been tested with version 2.)

Shut down the cluster with

```bash
ray down cluster_aws.yml --yes --verbose
```

# GCP Cluster

Create the cluster with

```bash
ray up cluster_gcp.yml --yes --verbose
```

Allow use of `ray.init("auto")` and get access to the dashboard with

```bash
ray dashboard cluster_gcp.yml
```

Submit jobs to the cluster with

```bash
ray job submit --runtime-env-json '{"working_dir":".","conda":"environment.yml"}' --no-wait --verbose -- python policy_inference_after_training.py
```

Shut down the cluster with

```bash
ray down cluster_gcp.yml --yes --verbose
```