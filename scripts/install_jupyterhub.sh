#!/usr/bin/env bash

### XXX: This script is in-development and not intended to work
### XXX: You probably shouldn't run this.
### INFO: https://zero-to-jupyterhub.readthedocs.io/en/latest/setup-jupyterhub.html

export secret=`openssl rand -hex 32`
export config_file=config.yaml
export RELEASE=jhub
export NAMESPACE=jhub


echo "proxy:" > ${config_file}
echo "  secretToken: \"${secret}\"" >> ${config_file}

helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/
helm repo update

helm upgrade --install $RELEASE jupyterhub/jupyterhub \
  --namespace $NAMESPACE  \
  --version=0.8.0-beta.1 \
  --values config.yaml