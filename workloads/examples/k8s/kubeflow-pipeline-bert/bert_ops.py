#!/usr/bin/env python3
import kfp.dsl as dsl
from kubernetes import client as k8s_client


__BERT_CONTAINER_VERSION__ = 'nvcr.io/nvidian/sae/atetelman-bert-kubeflow:v0.1'


class ObjectDict(dict):
  def __getattr__(self, name):
    if name in self:
      return self[name]
    else:
      raise AttributeError("No such attribute: " + name)


class BertCreateVolume(dsl.ResourceOp):
  def __init__(self, name, pv_name):
    super(BertCreateVolume, self).__init__(
      k8s_resource=k8s_client.V1PersistentVolumeClaim(
      api_version="v1", kind="PersistentVolumeClaim",
      metadata=k8s_client.V1ObjectMeta(name=pv_name),
      spec=k8s_client.V1PersistentVolumeClaimSpec(
          access_modes=['ReadWriteMany'], resources=k8s_client.V1ResourceRequirements(
              requests={'storage': '2000Gi'}),
          storage_class_name="nfs-client")),
      action='apply',
      name=name
      )
    name=name


class BertPrepData(dsl.ContainerOp):
  '''Data is downloaded to /workspace/bert/data/<model>'''
  def __init__(self, name, storage):
    cmd = ["/bin/bash", "-cx"]
    arguments = ["cd data; cp -a . " + storage]

    super(BertPrepData, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name

  
class BertPrepResults(dsl.ContainerOp):
  '''Data is downloaded to /workspace/bert/data/<model>'''
  def __init__(self, name, storage):
    cmd = ["/bin/bash", "-cx"]
    arguments = ["date >> " + storage + "runlog.log"]

    super(BertPrepResults, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BertStart(dsl.ContainerOp):
  '''Data is downloaded to /workspace/bert/data/<model>'''
  def __init__(self, name, model_type):
    cmd = ["/bin/bash", "-cx"]
    arguments = ["echo Starting: " + str(model_type) + "; df -h; du -sh /*; ls -latrh *; ls -latrh */*; date >> results/runlog.log; date >> data/runlog.log; echo Done"]

    super(BertStart, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BertDataDownload(dsl.ContainerOp):
  '''Data is downloaded to /workspace/bert/data/<model>'''
  def __init__(self, name, model_type):
    cmd = ["/bin/bash", "-x"]
    arguments = ["data/create_datasets_from_start.sh"]

    super(BertDataDownload, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BioBertDataDownload(dsl.ContainerOp):
  '''Data is downloaded to /workspace/bert/data/<model>'''
  def __init__(self, name, model_type):
    cmd = ["/bin/bash", "-x"]
    arguments = ["data/create_biobert_datasets_from_start.sh"]

    super(BioBertDataDownload, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BertCheckpointDownload(dsl.ContainerOp):
  '''Results go into /workspace/bert/results/<model>'''
  def __init__(self, name, model_type):
    cmd = ["/bin/bash", "-cx"]
    arguments = ["echo No Operation"]

    super(BertCheckpointDownload, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BioBertCheckpointDownload(dsl.ContainerOp):
  '''Results go into /workspace/bert/results/<model>'''
  def __init__(self, name, model_type):
    cmd = ["/bin/bash", "-cx"]
    arguments = ["echo No Operation"]

    super(BioBertCheckpointDownload, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BertPretrainingPhaseBoth(dsl.ContainerOp):
  '''Results go into /workspace/bert/results/<model>'''
  def __init__(self, name, model_type, batch_1, batch_2, lr_1,
                     lr_2, precision, use_xla, num_gpu,
                     warmup_1, warmup_2, train_both, accumulation_1,
                     accumulation_2, save_checkpoint_steps, bert_model):
    cmd = ["/bin/bash", "-x"]

    args = ["scripts/run_pretraining_lamb.sh", batch_1, batch_2, lr_1, lr_2,
                 precision, use_xla, num_gpu, warmup_1, warmup_2, train_both, 
                 save_checkpoint_steps, accumulation_1, accumulation_2, bert_model]
    arguments = [" ".join([str(arg) for arg in args])]

    super(BertPretrainingPhaseBoth, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BioBertPretrainingPhase1(dsl.ContainerOp):
  '''Results go into /workspace/bert/results/<model>'''
  def __init__(self, name, model_type, batch_1,
                     lr_1, cased, precision, use_xla, num_gpu,
                     warmup_1, train_1, accumulation_1,
                     save_checkpoint_steps, batch_1_eval):
    cmd = ["/bin/bash", "-x"]
    args = ["biobert/scripts/run_pretraining-pubmed_base_phase_1.sh", batch_1, lr_1, cased, precision, use_xla, num_gpu,
                  warmup_1, train_1, accumulation_1, save_checkpoint_steps, batch_1_eval]

    arguments = [" ".join([str(arg) for arg in args])]

    super(BioBertPretrainingPhase1, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BioBertPretrainingPhase2(dsl.ContainerOp):
  ''' Standard BERT has no phase 2'''
  def __init__(self, name, model_type, batch_2,
                     lr_2, cased, precision, use_xla, num_gpu,
                     warmup_2, train_2, accumulation_2,
                     save_checkpoint_steps, batch_2_eval, checkpoint_1):
    cmd = ["/bin/bash", "-x"]
    args = ["biobert/scripts/run_pretraining-pubmed_base_phase_2.sh", checkpoint_1, batch_2, lr_2, cased, precision, use_xla, num_gpu,
                 warmup_2, train_2, accumulation_2, save_checkpoint_steps, batch_2_eval]
    arguments = [" ".join([str(arg) for arg in args])]

    super(BioBertPretrainingPhase2, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BioBertChemFineTuning(dsl.ContainerOp):
  def __init__(self, name, task_type, checkpoint_pretrain, batch_tune,
                     lr_tune, cased, precision, use_xla,
                     num_gpu, seq_length, bert_model, batch_eval, epochs_eval,
                     doc_stride, squad_version):
    cmd = ["/bin/bash", "-x"]
    args = ["biobert/scripts/ner_bc5cdr-chem.sh", checkpoint_pretrain, batch_tune, lr_tune, cased, precision,
                 use_xla, num_gpu, seq_length, bert_model, batch_eval, epochs_eval]
    arguments = [" ".join([str(arg) for arg in args])]


    super(BioBertChemFineTuning, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BioBertDiseaseFineTuning(dsl.ContainerOp):
  def __init__(self, name, task_type, checkpoint_pretrain, batch_tune,
                     lr_tune, cased, precision, use_xla,
                     num_gpu, seq_length, bert_model, batch_eval, epochs_eval,
                     doc_stride, squad_version):
    cmd = ["/bin/bash", "-x"]
    args = ["biobert/scripts/ner_bc5cdr-disease.sh", checkpoint_pretrain, batch_tune, lr_tune, cased, precision,
                 use_xla, num_gpu, seq_length, bert_model, batch_eval, epochs_eval]
    arguments = [" ".join([str(arg) for arg in args])]

    super(BioBertDiseaseFineTuning, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BioBertRelationFineTuning(dsl.ContainerOp):
  def __init__(self, name, task_type, checkpoint_pretrain, batch_tune,
                     lr_tune, cased, precision, use_xla,
                     num_gpu, seq_length, bert_model, batch_eval, epochs_eval,
                     doc_stride, squad_version):
    cmd = ["/bin/bash", "-x"]
    args = ["biobert/scripts/rel_chemprot.sh", checkpoint_pretrain, batch_tune, lr_tune, cased, precision, 
                 use_xla, num_gpu, seq_length, bert_model, batch_eval, epochs_eval]
    arguments = [" ".join([str(arg) for arg in args])]

    super(BioBertRelationFineTuning, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BertGlueFineTuning(dsl.ContainerOp):
  def __init__(self, name, task_type, checkpoint_pretrain, batch_tune,
                     lr_tune, cased, precision, use_xla,
                     num_gpu, seq_length, bert_model, batch_eval, epochs_eval,
                     doc_stride, squad_version):
    # TODO:
    # cmd = ["/bin/bash", "-x", "scripts/run_glue.sh"]
    #  <task_name> <batch_size_per_gpu> <learning_rate_per_gpu> <precision> <use_xla> <num_gpu>
    #  <seq_length> <doc_stride> <bert_model> <epochs> <warmup_proportion> <checkpoint>
    cmd = ["/bin/bash", "-cx"]
    arguments = ["echo No-op"]

    super(BertGlueFineTuning, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BertSquadFineTuning(dsl.ContainerOp):
  def __init__(self, name, task_type, checkpoint_pretrain, batch_tune,
                     lr_tune, cased, precision, use_xla,
                     num_gpu, seq_length, bert_model, batch_eval, epochs_eval,
                     doc_stride, squad_version):
    cmd = ["/bin/bash", "-x"]
    args = ["scripts/run_squad.sh", batch_tune, lr_tune, precision, use_xla, num_gpu,
                 seq_length, doc_stride, bert_model, squad_version, checkpoint_pretrain, epochs_eval]
    arguments = [" ".join([str(arg) for arg in args])]


    super(BertSquadFineTuning, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BioBertEvaluate(dsl.ContainerOp):
  '''Data is downloaded to /workspace/bert/data/<model>'''
  def __init__(self, name, model_type, task_type, checkpoint_final, bert_model, cased,
               precision, use_xla, batch_size, doc_stride, seq_length):
    cmd = ["/bin/bash", "-x"]
    args = ["biobert/scripts/run_biobert_finetuning_inference.sh",
            task_type, checkpoint_final, bert_model, cased, precision,
                 use_xla, batch_size]
    arguments = [" ".join([str(arg) for arg in args])]

    super(BioBertEvaluate, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BertEvaluate(dsl.ContainerOp):
  '''Data is downloaded to /workspace/bert/data/<model>'''
  def __init__(self, name, model_type, task_type, checkpoint_final, bert_model, cased,
               precision, use_xla, batch_size, doc_stride, seq_length):
    cmd = ["/bin/bash", "-x"]
    args = ["scripts/run_squad_inference.sh", checkpoint_final, batch_size, precision, use_xla, seq_length,
                 doc_stride, bert_model, batch_size]
    arguments = [" ".join([str(arg) for arg in args])]

    super(BertEvaluate, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BertDeploy(dsl.ContainerOp):
  '''Data is downloaded to /workspace/bert/data/<model>'''
  def __init__(self, name, model_type):
    cmd = ["/bin/bash", "-cx"]
    arguments = ["echo No Operation"]

    super(BertDeploy, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name


class BioBertDeploy(dsl.ContainerOp):
  '''Data is downloaded to /workspace/bert/data/<model>'''
  def __init__(self, name, model_type):
    cmd = ["/bin/bash", "-cx"]
    arguments = ["echo No Operation"]

    super(BioBertDeploy, self).__init__(
      name=name,
      image=__BERT_CONTAINER_VERSION__,
      command=cmd,
      arguments=arguments,
      file_outputs={}
      )
    name=name
