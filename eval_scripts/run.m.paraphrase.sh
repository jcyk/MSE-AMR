set -e
#--drop_parentheses

gpu=$1
data=$2
parser=$3
for model in bert-base-uncased; do
another=${data}amr_16my-sup-simcse-${model}
prefix=debug.m.sup.paraphrase
amr=backup/${parser}
out=${prefix}.${parser}.${another}
rm -f ${out}

for j in 1 2 3 4 5; do
tmp_out=${out}.${j}
OUTPUT_DIR0=sentence-transformers/paraphrase-xlm-r-multilingual-v1
OUTPUT_DIR1=result/${another}${j}

echo "=========${another}${j}=========" >>  ${out}
CUDA_VISIBLE_DEVICES=${gpu} python3 evaluation.py --combine_method cat --normalize  --model_name_or_path ${OUTPUT_DIR0}:${OUTPUT_DIR1} --task_set msts --pooler avg:cls --use_amr --path_to_amr ${amr}/sts-test-pred.txt > ${tmp_out}
tail ${tmp_out}  >> ${out}
echo "=======================" >> ${out}
rm -f ${tmp_out}

done
done
