set -e

MODEL_NAME=$1

current_dir=$(pwd)
cd FastChat/fastchat/llm_judge
CUDA_VISIBLE_DEVICES=0 python gen_model_answer.py --model-path ${current_dir}/output/${MODEL_NAME} --model-id ${MODEL_NAME} --bench-name musique-ao --max-new-token 512 & \
CUDA_VISIBLE_DEVICES=1 python gen_model_answer.py --model-path ${current_dir}/output/${MODEL_NAME} --model-id ${MODEL_NAME} --bench-name musique-cot --max-new-token 512 & \
CUDA_VISIBLE_DEVICES=2 python gen_model_answer.py --model-path ${current_dir}/output/${MODEL_NAME} --model-id ${MODEL_NAME} --bench-name musique-coc --max-new-token 512
wait $!
cd ${current_dir}
echo "MuSiQue CoC"
python eval_multihop.py --bench-name musique-coc
echo "MuSiQue CoT"
python eval_multihop.py --bench-name musique-cot
echo "MuSiQue AO"
python eval_multihop.py --bench-name musique-ao