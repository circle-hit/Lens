

export PATH="/home/wxzhao/anaconda3/bin:$PATH"                                   
conda init       
source activate
conda activate gjh_ejb

CONFIG_PATH=Probing/configs/single/llama3-instruct-8b.ini

python Probing/main.py --config $CONFIG_PATH
