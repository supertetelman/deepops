#!/usr/bin/env bash

# Ensure we start in the correct working directory
ROOT_DIR="${SCRIPT_DIR}/../.."
cd "${ROOT_DIR}" || exit 1

# Allow overriding config dir to look in
DEEPOPS_CONFIG_DIR=${DEEPOPS_CONFIG_DIR:-"${ROOT_DIR}/config"}

# Create namespace and empty NFS PVC
kubectl create namespace deepops-triton
kubectl apply -f workloads/examples/k8s/triton-horizontal-scale/pvc.yml -n deepops-triton

# Create Sample Model Repo
pushd

cd submodules
git clone https://github.com/triton-inference-server/server.git
cd server/docs/examples
./fetch_models.sh
cp -r model_repository/ /export/deepops_nfs/deepops-triton-triton-claim-pvc-*/

# XXX: These fail to load on the included Triton version
rm -rf /export/deepops_nfs/deepops-triton-triton-claim-pvc-*/model_repository/simple_identity
rm -rf /export/deepops_nfs/deepops-triton-triton-claim-pvc-*/model_repository/inception_graphdef

popd

# Deploy Triton Server
pushd
cd submodules/tritoninferenceserver/ && helm install nvidia . --namespace deepops-triton
popd

# Deploy special monitoring stack
export PROMETHEUS_YAML_CONFIG=workloads/examples/k8s/triton-horizontal-scale/config/helm/monitoring.yml
./scripts/k8s/deploy_monitoring.sh -d && sleep 10
./scripts/k8s/deploy_monitoring.sh

# Deploy additional Prometheus Adapter and customer metrics
helm install prometheus-community/prometheus-adapter    --namespace monitoring    --generate-name    --values config/helm/prometheus-adapter.values

# Verify metrics are available
echo 'Run command to verify metrics

kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/deepops-triton/pods/*/nv_inference_queue_duration_us" | jq .'
