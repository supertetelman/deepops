#!/usr/bin/env bash

### XXX: This script is in-development and not intended to work
### XXX: You probably shouldn't run this.
### INFO: https://www.kubeflow.org/docs/started/getting-started/
### INFO: https://www.kubeflow.org/docs/other-guides/accessing-uis/

export KUBEFLOW_SOURCE=/home/ubuntu/kf_src
export KUBEFLOW_TAG=v0.4.1
export KFAPP=kfapp
export NAMESPACE=kubeflow

mkdir ${KUBEFLOW_SRC}
cd ${KUBEFLOW_SRC}

curl https://raw.githubusercontent.com/kubeflow/kubeflow/${KUBEFLOW_TAG}/scripts/download.sh | bash

${KUBEFLOW_SRC}/scripts/kfctl.sh init ${KFAPP} --platform none
cd ${KFAPP}
${KUBEFLOW_SRC}/scripts/kfctl.sh generate k8s
${KUBEFLOW_SRC}/scripts/kfctl.sh apply k8s

kubectl port-forward svc/ambassador -n ${NAMESPACE} 8080:80