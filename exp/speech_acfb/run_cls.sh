stage=2
stop_stage=3
audiocategory=speech
classifier=BLSTM_deltas
cd ../../
. parse_options.sh

tag=$1

augpath='none'
emp=feats  
aug_for_cls=false 
preTrdir_tag=AugPreTrained

feature_dir=audio_flac 
augfeature_dir=augmentation/${audiocategory}
datadir=data/$outerid/
feats_config='conf/feature.conf'
timestr=$(date '+%Y%m%d%H%M%S') 

 outdir=${audiocategory}/${outerid}/results_${classifier}/
 mkdir -p $outdir

if [ $aug_for_cls = true ]; then
    augtag=AugCls 
    echo "Augmentation allowed for classifier training."
else
    augtag=AugPreTrained
    echo "Augmentation only for FB training."
fi
if [ $augpath = 'none' ]; then
    augtag=""
fi
echo "augtag: $augtag"
preTrdir_tag=$augtag


 #=============================== Put the params here ======================================

 train_config='conf/train_config'
 model_config='conf/model_config'
 nettype='BLSTM_segment'
 acfb_LR=0.01
 cls_LR=0.0001
 l2_lambda=0.001
 activation='tanh'
 seed=42
 n_bins=64
 acfb_init='rand1'
 n_files_load=150
 n_epochs=10
 bsz=128
 relev_type='adaptiveWt'
 optim='adam'
 wdecay=0.0001
 exp=$tag
 result_folder=$audiocategory/${emp}_results_${nettype}_${exp}${augtag}_acfb${acfb_LR}_cls${cls_LR}_lambda${l2_lambda}_${activation}_init${acfb_init}_sch${schdlr}
 echo $result_folder

if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    echo "======== Filter bank learning phase ========="


    for outerid in $(seq 0 4);do

        python local/train_aug_V2.py \
     -c $train_config \
     -m $model_config \
     -f $feats_config \
     --featsfil data/${audiocategory}_vad.scp \
     --trainfil data/${outerid}/dev \
     --valfil data/${outerid}/test \
     --result_folder $result_folder/$outerid/ \
     --augdir $augpath \
     --num_epochs $n_epochs \
     --batch_size 8 \
     --seed $seed \
     --max_lr $cls_LR \
     --acfb_lr 0.01 \
     --num_freq_bin $n_bins \
     --acfb_init $acfb_init \
     --mean_var_norm \
     --optim_method $optim \
     --wdecay $wdecay \
     --use_scheduler \
     --LBaseline \
     --random_net \
     --deltas \
     --last_model_path none \
     --num_files_load $n_files_load \
     --l2_lambda $l2_lambda \
     --net_activation $activation \

    done

   echo "stage 2 completed"

fi 


if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then

        echo "==== Classifier with preTrained AcFB  ====="

        wdecay=0.0001 
        datetime=$timestr
        
        preTr_result_folder=$result_folder
        

        echo  "result_folder=$result_folder"
        echo "using fb from $preTr_result_folder"
        if [ $aug_for_cls = false ]; then 
            augpath="none"
        fi

    for outerid in $(seq 0 4);do

        python local/train_aug_V2.py \
     -c $train_config \
     -m $model_config \
     -f $feats_config \
     --featsfil data/${audiocategory}_vad.scp \
     --trainfil data/${outerid}/dev \
     --valfil data/${outerid}/test \
     --result_folder $result_folder/$outerid/ \
     --augdir $augpath \
     --num_epochs $n_epochs \
     --batch_size $bsz \
     --seed $seed \
     --max_lr $cls_LR \
     --acfb_lr $acfb_LR \
     --num_freq_bin $n_bins \
     --acfb_init $acfb_init \
     --mean_var_norm \
     --optim_method $optim \
     --wdecay $wdecay \
     --use_scheduler \
     --acfb_preTrained \
     --random_net \
     --deltas \
     --last_model_path ${preTr_result_folder}/checkpoint/final_model.pth \
     --num_files_load $n_files_load \
     --l2_lambda $l2_lambda \
     --net_activation $activation \
     --use_relWt \
     --relevance_type $relev_type \


     
        for model in best_frame_auc_model;do

            datetime=$timestr
            #result_folder=$audiocategory/${emp}_results_${nettype}_${exp}${augtag}_acfb${acfb_LR}_cls${cls_LR}_lambda${l2_lambda}_${activation}_init${acfb_init}_sch${schdlr}
            echo "==== infering on $result_folder ===" 
            
            python local/infer.py -c $train_config -f $feats_config -m $result_folder/${outerid}/ClsCheckpoint/${model}.pth -i data/${outerid}/test_${audiocategory}.scp -o $result_folder/${outerid}/${model}_test_scores.txt \
            --featsfil data/${audiocategory}_vad.scp \
            --model_config $model_config \
            --acfb_preTrained \
            --mean_var_norm \
            --deltas \
            --use_relWt \
            --relevance_type $relev_type 

            python local/scoring.py -r data/${outerid}/test -t $result_folder/${outerid}/${model}_test_scores.txt -o $result_folder/${outerid}/${model}_test_results.pkl

        done
   done

   echo "stage 3 completed"
fi


echo "Done!!!"

