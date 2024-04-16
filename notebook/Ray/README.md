# Ray

## Table of Contents

- [Ray](#ray)
  - [Table of Contents](#table-of-contents)
  - [GCP Setup](#gcp-setup)

## GCP Setup

- [Github repo](https://github.com/ray-project/ray/tree/master/python/ray/autoscaler/gcp)
- Simple config

```yaml
auth:
  ssh_user: ubuntu
cluster_name: minimal
provider:
  availability_zone: us-west1-a
  project_id: null # TODO: set your GCP project ID here
  region: us-west1
  type: gcp
```
