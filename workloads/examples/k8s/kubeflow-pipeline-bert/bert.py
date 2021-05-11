#!/usr/bin/env python3
'''
Cannot dynamically assign GPU counts: https://github.com/kubeflow/pipelines/issues/1956

    # TODO: Parse evaulation and trigger more tuning
    # TODO: Option to train multiple BERT models at once
'''
import bert_ops
import kfp.dsl as dsl
from kubernetes import client as k8s_client

@dsl.pipeline(
    name='bertPipeline',
    description='BERT and BioBERT Kubeflow Pipeline'
)
def bert_pipeline(model_type,
    train_1,train_2,train_both,cased,doc_stride,task_type,
    epochs_tune,lr_1,lr_2,lr_tune,batch_1,batch_2,batch_tune,
    batch_1_eval,batch_2_eval,batch_tune_eval,batch_eval,precision,use_xla,
    num_gpu,warmup_1,warmup_2,accumulation_1,accumulation_2,steps,
    bert_model,squad_version,seq_length,
    checkpoint_steps):

    op_dict = {}

    # Hardcoded data/paths, see https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/data/create_datasets_from_start.sh
    prep_dir = "/bert-prep-copy" # Used to temporarily copy the /workspace/bert/data files into the NFS
    results_dir = "/workspace/bert/results/" # The BERT scripts have this hardcoded
    data_dir = "/workspace/bert/data/" # The BERT scripts have this hardcoded
    pv_data_name = "bert-data"
    pv_results_name = "bert-results"

    # TODO: Checkpoint dirs
    checkpoint_1 = 'TODO'
    checkpoint_pretrain = 'TODO'
    checkpoint_final = ' TODO'

    # Initial volume creations
    op_dict['bert_data_volume_creation'] = bert_ops.BertCreateVolume('bert_data_volume_creation', pv_data_name)
    op_dict['bert_results_volume_creation'] = bert_ops.BertCreateVolume('bert_results_volume_creation', pv_results_name)

    # Common Operations
    op_dict['bert_prep_data'] = bert_ops.BertPrepData('bert_prep_data', prep_dir)
    op_dict['bert_prep_results'] = bert_ops.BertPrepResults('bert_prep_results', prep_dir)
    op_dict['bert_start'] = bert_ops.BertStart('bert_start', model_type)

    with dsl.Condition(model_type == 'bert'):
        op_dict['bert_data_download'] = bert_ops.BertDataDownload('bert_data_download', model_type)

        op_dict['bert_checkpoint_download'] = bert_ops.BertCheckpointDownload('bert_checkpoint_download', model_type)

        op_dict['bert_pretrain_both'] = bert_ops.BertPretrainingPhaseBoth('bert_pretrain_both', model_type, batch_1, batch_2, lr_1,
                                                                         lr_2, precision, use_xla, num_gpu, warmup_1, warmup_2, train_both,
                                                                         accumulation_1, accumulation_2, checkpoint_steps,
                                                                         bert_model).set_gpu_limit(1, vendor = "nvidia")

        with dsl.Condition(task_type == 'squad'):
            op_dict['bert_squad_finetune'] = bert_ops.BertSquadFineTuning('bert_squad_finetune', task_type, checkpoint_pretrain, batch_tune,
                                                          lr_tune, cased, precision, use_xla, num_gpu, seq_length,
                                                          bert_model, batch_tune_eval, epochs_tune, doc_stride,
                                                          squad_version).set_gpu_limit(1, vendor = "nvidia")

        with dsl.Condition(task_type == 'glue'):
            op_dict['bert_glue_finetune'] = bert_ops.BertGlueFineTuning('bert_glue_finetune', task_type, checkpoint_pretrain, batch_tune,
                                                          lr_tune, cased, precision, use_xla, num_gpu, seq_length,
                                                          bert_model, batch_tune_eval, epochs_tune, doc_stride,
                                                          squad_version).set_gpu_limit(1, vendor = "nvidia")

        op_dict['bert_evaluate'] = bert_ops.BertEvaluate('bert_evaluate', model_type, task_type, checkpoint_final, bert_model, 
                                                        cased, precision, use_xla, batch_eval, doc_stride,
                                                        seq_length).set_gpu_limit(1, vendor = "nvidia")

        op_dict['bert_deploy'] = bert_ops.BertDeploy('bert_deploy', model_type)

    with dsl.Condition(model_type == 'biobert'):
        op_dict['bio_bert_data_download'] = bert_ops.BioBertDataDownload('bio_bert_data_download', model_type)

        op_dict['bio_bert_checkpoint_download'] = bert_ops.BioBertCheckpointDownload('bio_bert_checkpoint_download', model_type)

        op_dict['bio_bert_pretrain_1'] = bert_ops.BioBertPretrainingPhase1('bio_bert_pretrain_1', model_type, batch_1, lr_1,
                                                                   cased, precision, use_xla, num_gpu, warmup_1, train_1,
                                                                   accumulation_1, checkpoint_steps,
                                                                   batch_1_eval).set_gpu_limit(1, vendor = "nvidia")

        op_dict['bio_bert_pretrain_2'] = bert_ops.BioBertPretrainingPhase2('bio_bert_pretrain_2', model_type, batch_2, lr_2,
                                                                   cased, precision, use_xla, num_gpu, warmup_2, train_2,
                                                                   accumulation_2, checkpoint_steps, batch_2_eval,
                                                                   checkpoint_1).set_gpu_limit(1, vendor = "nvidia")
        with dsl.Condition(task_type == 'chem'):
            op_dict['bio_bert_chem_finetune'] = bert_ops.BioBertChemFineTuning('bio_bert_chem_finetune', task_type, checkpoint_pretrain, batch_tune,
                                                          lr_tune, cased, precision, use_xla, num_gpu, seq_length,
                                                          bert_model, batch_tune_eval, epochs_tune, doc_stride,
                                                          squad_version).set_gpu_limit(1, vendor = "nvidia")
        with dsl.Condition(task_type == 'disease'):
            op_dict['bio_bert_disease_finetune'] = bert_ops.BioBertDiseaseFineTuning('bio_bert_disease_finetune', task_type, checkpoint_pretrain, batch_tune,
                                                          lr_tune, cased, precision, use_xla, num_gpu, seq_length,
                                                          bert_model, batch_tune_eval, epochs_tune, doc_stride,
                                                          squad_version).set_gpu_limit(1, vendor = "nvidia")
        with dsl.Condition(task_type == 'relation'):
            op_dict['bio_bert_relation_finetune'] = bert_ops.BioBertRelationFineTuning('bio_bert_relation_finetune', task_type, checkpoint_pretrain, batch_tune,
                                                          lr_tune, cased, precision, use_xla, num_gpu, seq_length,
                                                          bert_model, batch_tune_eval, epochs_tune, doc_stride,
                                                          squad_version).set_gpu_limit(1, vendor = "nvidia")

  
        op_dict['bio_bert_evaluate'] = bert_ops.BioBertEvaluate('bio_bert_evaluate', model_type, task_type, checkpoint_final, bert_model, 
                                                        cased, precision, use_xla, batch_eval, doc_stride,
                                                        seq_length).set_gpu_limit(1, vendor = "nvidia")

        op_dict['bio_bert_deploy'] = bert_ops.BioBertDeploy('bio_bert_deploy', model_type)

    # Create initial volumes if they do not exist
    op_dict['bert_results_volume_creation'].after(op_dict['bert_data_volume_creation'])

    # Save off data and results scripts into the NFS volumes that over-rode them potentially
    op_dict['bert_prep_data'].after(op_dict['bert_results_volume_creation'])
    op_dict['bert_prep_results'].after(op_dict['bert_results_volume_creation'])

    # Print out some basic debug information before training 
    op_dict['bert_start'].after(op_dict['bert_prep_data'])
    op_dict['bert_start'].after(op_dict['bert_prep_results'])

    # BERT training
    op_dict['bert_data_download'].after(op_dict['bert_start'])
    op_dict['bert_checkpoint_download'].after(op_dict['bert_data_download'])
    op_dict['bert_pretrain_both'].after(op_dict['bert_checkpoint_download'])
    op_dict['bert_squad_finetune'].after(op_dict['bert_pretrain_both'])
    op_dict['bert_glue_finetune'].after(op_dict['bert_pretrain_both'])
    op_dict['bert_evaluate'].after(op_dict['bert_glue_finetune'])
    op_dict['bert_evaluate'].after(op_dict['bert_squad_finetune'])
    op_dict['bert_deploy'].after(op_dict['bert_evaluate'])

    # BioBERT training
    op_dict['bio_bert_data_download'].after(op_dict['bert_start'])
    op_dict['bio_bert_checkpoint_download'].after(op_dict['bio_bert_data_download'])
    op_dict['bio_bert_pretrain_1'].after(op_dict['bio_bert_checkpoint_download'])
    op_dict['bio_bert_pretrain_2'].after(op_dict['bio_bert_pretrain_1'])
    op_dict['bio_bert_chem_finetune'].after(op_dict['bio_bert_pretrain_2'])
    op_dict['bio_bert_disease_finetune'].after(op_dict['bio_bert_pretrain_2'])
    op_dict['bio_bert_relation_finetune'].after(op_dict['bio_bert_pretrain_2'])
    op_dict['bio_bert_evaluate'].after(op_dict['bio_bert_chem_finetune'])
    op_dict['bio_bert_evaluate'].after(op_dict['bio_bert_disease_finetune'])
    op_dict['bio_bert_evaluate'].after(op_dict['bio_bert_relation_finetune'])
    op_dict['bio_bert_deploy'].after(op_dict['bio_bert_evaluate'])

    # In different operatios we want to mount the results or data volumes in different modes
    for name, container_op in op_dict.items():
        if type(container_op) == bert_ops.BertCreateVolume:
            continue
        elif name == 'bert_prep_data': # Prep Ops need to mount Volume in special directory to not overwrite container files
            container_op.add_volume(k8s_client.V1Volume(persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                claim_name=pv_data_name, read_only=False), name=pv_data_name))  
            container_op.add_volume_mount(k8s_client.V1VolumeMount(
                mount_path=prep_dir,
                name=pv_data_name)) # Replace _ with - for K8s compatibility reasons
            continue
        elif name == 'bert_prep_results': # Prep Ops need to mount Volume in special directory to not overwrite container files
            container_op.add_volume(k8s_client.V1Volume(persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                claim_name=pv_results_name, read_only=False), name=pv_results_name)) 
            container_op.add_volume_mount(k8s_client.V1VolumeMount(
                mount_path=prep_dir,
                name=pv_results_name))
            continue
        elif name == "bert_data_download": # For data safety reasons, make the data volume RW only when downloading data
            container_op.add_volume(k8s_client.V1Volume(persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                claim_name=pv_data_name, read_only=False), name=pv_data_name))
            container_op.add_volume_mount(k8s_client.V1VolumeMount(
                mount_path=data_dir,
                name=pv_data_name,
                read_only=False))

            container_op.add_volume(k8s_client.V1Volume(persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                claim_name=pv_results_name, read_only=False), name=pv_results_name)) 
            container_op.add_volume_mount(k8s_client.V1VolumeMount(
                mount_path=results_dir,
                name=pv_results_name))
        else:
            container_op.add_volume(k8s_client.V1Volume(persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                claim_name=pv_data_name, read_only=False), name=pv_data_name))
            container_op.add_volume_mount(k8s_client.V1VolumeMount(
                mount_path=data_dir,
                name=pv_data_name,
                read_only=True))

            container_op.add_volume(k8s_client.V1Volume(persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                claim_name=pv_results_name, read_only=False), name=pv_results_name)) 
            container_op.add_volume_mount(k8s_client.V1VolumeMount(
                mount_path=results_dir,
                name=pv_results_name))


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(bert_pipeline, __file__ + '.tar.gz')
