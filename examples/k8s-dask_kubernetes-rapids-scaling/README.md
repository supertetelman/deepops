# Scaling a Machine Learning Workload Using RAPIDS, Dask, and Kubernetes

## Introduction

RAPIDS provides a suite of open source software libraries for doing data science on GPUs. It's often used in conjunction with Dask, a Python framework for running parallel computing jobs. Both these tools can be used either in a containerized workflow, often using Kubernetes, or on "bare metal" with no containers, often using a shared HPC cluster.

In this example, we will be working our way up to a machine learning job that can run on dozens of GPUs or nodes spread out across our cluster. We will achieve this by combing the parallelizability of RAPIDS and Dask with the flexibility and scalability of Kubernetes.

In the [example notebook](dask.ipynb) we will introduce a real-life machine learning problem, follow a typcial data scientist development workflow to build an initial model and do pre-liminary scoping and discovery of the problem, and we will then train a model without Dask, with Dask on a single node, and then with Dask on multiple nodes.

By the end of the guide, you should have an environment setup to replicate our results as-well as the ability to use your kubernetes_dask cluster to build your own machine learning applications.

## Requirements

In order to run through this guide you will need at least 1 node with at least 2 V100 or other data-center grade GPUs. Ideally you will have multiple nodes each equipped with at least 4 GPUs.

## Deployment steps

1. Deploy a Kubernetes cluster.

Follow the steps listed in the [Kubernetes Guide](../../docs/kubernetes-cluster.md). Skip the Optional Components, we will be running through the required components later.

Verify the cluster is running by running
```sh
kubectl run gpu-test --rm -t -i --restart=Never --image=nvidia/cuda --limits=nvidia.com/gpu=1 -- nvidia-smi
```
   > Note: You should see nvidia-smi output with a single GPU

   TODO: add the output here

2. Deploy Persistant Storage

Follow the [steps to deploy persistant storage](../../docs/kubernetes-cluster.md#persistent-storage).

This persistant storage is primarily meant for storing configuration files, notebooks, and small files used for development. Models, data, and important files should be stored in a seperate persistant network storage device, the setup of which is not included this guide.

3. Deploy Kubeflow

At the the end of this step you should be given connectivity information for the Kubeflow dashboard. Copy/paste this url into your browser and verify you are able to connect into Kubeflow.

   > Note: If you ever lose the connectivity information you can run `kubectl get svc -n kubeflow` and look at the `NodePort` address used by the `centraldashboard`.

4. Deploy Dask using Helm

There are multiple different ways to deploy and use Dask within Kubernetes. In this guide we will abstract some of this away and focus on demonstrating how to scale with the python library `dask_kubernetes`.

This script can be used to deploy Dask outside of Kubeflow. Running this script standalone allows us to completely delete and re-deploy a Dask cluster, but because we do not want to do that and we want to deploy alongside the `kubeflow` namespace we pass a few additional parameters to the script. This is purely how our deployment scripts are implemented and not very important to learning Dask.

```sh
export RAPIDS_SKIP_KUBERNETES=true
yes | ./scripts/k8s_deploy_rapids_dask.sh -n kubeflow
```

5. Launch a RAPIDS container through Kubeflow

At this point you should be all ready to log back into Kubeflow and get started with Dask. Connect to the centraldashboard, select the `Notebooks` section on the sidebar, select `new notebook`, and select the latest RAPIDS container from the drop down list. 

Select a suitable amount of CPU, memory, and 1 GPU for your container.

After clicking the start button it may take some time for the container image to download, when it completes click the connect button and follow the tutorial.

## Tutorial

## Performance Summary

## Additional Resources

## FAQ

