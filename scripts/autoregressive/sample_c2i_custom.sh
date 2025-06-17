#!/bin/bash
set -ex  # -e: 에러 시 종료, -x: 명령어 출력

# 기본값 설정 (parser의 default 값과 일치)
GPT_MODEL="GPT-B"
GPT_CKPT=""
GPT_TYPE="c2i"
VQ_MODEL="VQ-16"
VQ_CKPT=""
IMAGE_SIZE=384  # 기본값이 384임
IMAGE_SIZE_EVAL=256
DOWNSAMPLE_SIZE=16
TOP_K=0
TOP_P=1.0
TEMPERATURE=1.0
CFG_SCALE=1.5  # 기본값이 1.5임
LOCAL_SCALE=0.5  # 기본값이 0.5임
WINDOW_PARAM=64
WINDOW_TYPE="1d"
BATCH_SIZE=32  # 기본값이 32임
SEED=0

# 인수 파싱
ARGS=("$@")  # 원본 인수 백업
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpt-model)
            GPT_MODEL="$2"
            shift 2
            ;;
        --gpt-ckpt)
            GPT_CKPT="$2"
            shift 2
            ;;
        --gpt-type)
            GPT_TYPE="$2"
            shift 2
            ;;
        --vq-model)
            VQ_MODEL="$2"
            shift 2
            ;;
        --vq-ckpt)
            VQ_CKPT="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --image-size-eval)
            IMAGE_SIZE_EVAL="$2"
            shift 2
            ;;
        --downsample-size)
            DOWNSAMPLE_SIZE="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --cfg-scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        --local-guidance-scale)
            LOCAL_SCALE="$2"
            shift 2
            ;;
        --window-parameter)
            WINDOW_PARAM="$2"
            shift 2
            ;;
        --window-type)
            WINDOW_TYPE="$2"
            shift 2
            ;;
        --per-proc-batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --global-seed)
            SEED="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# torchrun 실행 (원래 인수 그대로 전달)
# 주의: 스크립트에서 이미 --vq-ckpt를 지정하고 있으므로 중복 방지 필요
torchrun \
--nnodes=1 --nproc_per_node=4 --node_rank=0 \
--master_port=12345 \
autoregressive/sample/sample_c2i_ddp.py \
"${ARGS[@]}"

# GPT checkpoint에서 모델 이름 추출 (예: ./pretrained_models/c2i_B_256.pt -> c2i_B_256)
if [ -n "$GPT_CKPT" ]; then
    CKPT_STRING_NAME=$(basename "$GPT_CKPT" .pt)
else
    CKPT_STRING_NAME="none"  # checkpoint가 없을 경우
fi

# 파일명 생성 (폴더명 패턴과 동일)
FOLDER_NAME="${GPT_MODEL}-${CKPT_STRING_NAME}-size-${IMAGE_SIZE}-size-${IMAGE_SIZE_EVAL}-${VQ_MODEL}-topk-${TOP_K}-topp-${TOP_P}-temperature-${TEMPERATURE}-cfg-${CFG_SCALE}-local-${LOCAL_SCALE}-window-${WINDOW_PARAM}-${WINDOW_TYPE}-batchsize-${BATCH_SIZE}-seed-${SEED}"

OUTPUT_FILE="samples/${FOLDER_NAME}.npz"

# 평가 실행
python3 evaluations/c2i/evaluator.py VIRTUAL_imagenet256_labeled.npz "$OUTPUT_FILE"

# 평가 후 정리
rm -rf "samples/${FOLDER_NAME}"
rm -f "$OUTPUT_FILE"