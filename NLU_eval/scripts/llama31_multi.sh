#!/bin/bash
#SBATCH -J NLU%j                           # 作业名为 test
#SBATCH -o logs/%j.out                   
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 4:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --mem 100g
#SBATCH --gres=gpu:a100-pcie-40gb:2        # 申请 4 卡 A100 80GB，如果只申请CPU可以删除本行

source ~/.bashrc
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

source /home/jhguo/venv/LLM/bin/activate

export all_proxy=172.20.240.220:7890


model_name_or_path=/home/jhguo/Downloads/Llama-3.1-8B-Instruct-hf
output_dir=/home/jhguo/Code/NLU_eval_for_upload/results/llama31_multi

if [ -d $output_dir ]; then
    echo "文件夹存在"
else
    mkdir $output_dir
fi

language_names=("zh" "ar" "en" "bn" "sw" "jp" )
dataset=m-mmlu
for language_name in "${language_names[@]}"
do
    echo "running on $dataset with language $language_name"
    python /home/jhguo/Code/NLU_eval_for_upload/src/main_no_match.py     --model_name_or_path $model_name_or_path     --dataset $dataset     --language_name $language_name     --output_dir $output_dir     --bs 8     --num 1000     --overwrite false
done

language_names=("zh" "en" "bn" "sw" "jp" "ar")
datasets=("xcopa" "xstory" "xwino")

for language_name in "${language_names[@]}"
do
    for dataset in "${datasets[@]}"
    do
        echo "running on $dataset with language $language_name"
        python /home/jhguo/Code/NLU_eval_for_upload/src/main_no_match.py         --model_name_or_path $model_name_or_path         --dataset $dataset         --language_name $language_name         --output_dir $output_dir         --bs 32         --num 1000         --overwrite false
    done
done

language_names=("ko")
datasets=("kmmlu")

for language_name in "${language_names[@]}"
do
    for dataset in "${datasets[@]}"
    do
        echo "running on $dataset with language $language_name"
        python /home/jhguo/Code/NLU_eval_for_upload/src/main_no_match.py         --model_name_or_path $model_name_or_path         --dataset $dataset         --language_name $language_name         --output_dir $output_dir         --bs 32         --num 1000         --overwrite false
    done
done
