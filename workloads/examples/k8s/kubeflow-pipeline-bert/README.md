# NGC Bert Kubeflow Pipeline

This Kubeflow pipeline is an implementation of the [NGC BERT](https://ngc.nvidia.com/catalog/resources/nvidia:bert_for_tensorflow/quickStartGuide) and [NGC BioBERT](https://ngc.nvidia.com/catalog/resources/nvidia:biobert_for_tensorflow/quickStartGuide) model scripts. Specifically this encapsulates the steps outlined in the getting started guides.

In order for this pipeline to work it is first necessary to build the `bert` container following the NGC steps and push it to a local Docker registry.


## Setup

* Kubeflow 
* NFS Client Provisioner, as installed in a default DeepOps deployment in [nfs-client-provisioner.yaml](../../playbooks/k8s-cluster/nfs-client-provisioner.yaml)

## Usage

### Compiling

### Uploading

### Running
