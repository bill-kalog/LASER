 #!/bin/sh

edir="/shared1/vasilis/datasets/sst/preprocessed"
log_dir="/shared1/vasilis/datasets/sst/log_dir"
model_folder="/shared1/vasilis/datasets/sst/models"
echo -e "\nTraining classifier (log files in ${log_dir})"

mkdir -p $log_dir
mkdir -p $model_folder



# classifier parameters


N=200
lr=0.001
wd=0.000
nhid="10 8"
nhid="500 300 100 50 20 10"
drop=0.2
seed=1
# bsize=12
bsize=256
# number of classes
nb_cl=5
# binary or fine
type="fine"

train_file=stsa.${type}.train.corpus.embeddings
# train_file=stsa.${type}.phrases.train.corpus.embeddings
val_file=stsa.${type}.dev.corpus.embeddings
# val_file=../../mathias_datasets/sentimentData_products.corpus.embeddings.sv
test_file=stsa.${type}.test.corpus.embeddings

# test_file=../../mathias_datasets/sentimentData_products.corpus.embeddings
# test_file=../../mathias_datasets/SEB_new.corpus.embeddings

# test_file=../../cmore/cmore_5.embeddings



train_file_labels=stsa.${type}.train.labels
# train_file_labels=stsa.${type}.phrases.train.labels
val_file_labels=stsa.${type}.dev.labels
# val_file_labels=../../mathias_datasets/sentimentData_products.labels.sv
test_file_labels=stsa.${type}.test.labels

# test_file_labels=../../mathias_datasets/sentimentData_products.labels
# test_file_labels=../../mathias_datasets/SEB_new.labels

# test_file_labels=../../cmore/cmore_5.labels

# lang=sv
lang=en


lf="${log_dir}/sst.${type}-28jan.log"

 
echo " - train on ${train_file}, dev on ${val_file}"
if [ ! -f ${lf} ] ; then
# CUDA_VISIBLE_DEVICES=3 python3 ${LASER}/source/sent_classif.py \
#   --gpu 3 --base-dir ${edir} \
#   --train ${train_file} \
#   --train-labels ${train_file_labels} \
#   --dev ${val_file} \
#   --dev-labels ${val_file_labels} \
#   --test ${test_file} \
#   --test-labels ${test_file_labels} \
#   --nb-classes ${nb_cl} \
#   --nhid ${nhid[@]} --dropout ${drop} --bsize ${bsize} \
#   --seed ${seed} --lr ${lr} --wdecay ${wd} --nepoch ${N} \
#   --lang ${lang} \
#   > ${lf}
CUDA_VISIBLE_DEVICES=3 python3 ${LASER}/source/sent_classif.py \
  --gpu 0 --base-dir ${edir} \
  --save "${model_folder}/sst_${type}_feb26" \
  --train ${train_file} \
  --train-labels ${train_file_labels} \
  --dev ${val_file} \
  --dev-labels ${val_file_labels} \
  --test ${test_file} \
  --test-labels ${test_file_labels} \
  --nb-classes ${nb_cl} \
  --nhid ${nhid[@]} --dropout ${drop} --bsize ${bsize} \
  --seed ${seed} --lr ${lr} --wdecay ${wd} --nepoch ${N} \
  --lang ${lang}
  "$@"
fi
