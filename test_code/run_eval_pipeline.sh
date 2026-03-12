#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_eval_pipeline.sh \
    --checkpoint /path/to/model_epoch_best.pth \
    --data-root /path/to/test_images_root \
    --model coco_ae_aligned_sr

Required:
  --checkpoint PATH   Model checkpoint (.pth)
  --data-root PATH    Test image root with domain subfolders (e.g., real, sd, mj...)
  --model NAME        Model name to register under test_code/weights/NAME

Optional:
  --patch-size VALUE          patch_size for config.yaml (e.g., 32, 256, null)
                              If omitted, auto-detect from existing config.yaml.
                              If detection is ambiguous, fallback default is 256.
  --csv-dir PATH              Where generated CSV files are saved
  --out-dir PATH              Where *_scored.csv and metrics_summary.csv are saved
  --weights-dir PATH          Base weights dir for main.py (default: <script_dir>/weights)
  --domains LIST              Comma list (default: real,kandinsky,latent_consistency,mj,pixelart,playground,sd)
  --max-files N|all           Per-domain CSV cap (default: 3000)
  --device DEV                main.py device arg (default: cuda:0)
  --cuda-visible-devices IDS  Optional CUDA_VISIBLE_DEVICES (e.g., 0)
  --ix N                      eval.py --ix (default: 1)
  --arch NAME                 config arch (default: res50nodown)
  --norm-type NAME            config norm_type (default: resnet)
  --python-bin BIN            Python executable (default: python)
  -h, --help                  Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKPOINT=""
DATA_ROOT=""
MODEL=""
PATCH_SIZE=""

CSV_DIR=""
OUT_DIR=""
WEIGHTS_DIR="$SCRIPT_DIR/weights"
DOMAINS_CSV="real,kandinsky,latent_consistency,mj,pixelart,playground,sd"
MAX_FILES="3000"
DEVICE="cuda:0"
CUDA_VISIBLE_IDS=""
IX="1"
ARCH="res50nodown"
NORM_TYPE="resnet"
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="${2:-}"; shift 2 ;;
    --data-root) DATA_ROOT="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --patch-size) PATCH_SIZE="${2:-}"; shift 2 ;;
    --csv-dir) CSV_DIR="${2:-}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    --weights-dir) WEIGHTS_DIR="${2:-}"; shift 2 ;;
    --domains) DOMAINS_CSV="${2:-}"; shift 2 ;;
    --max-files) MAX_FILES="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_IDS="${2:-}"; shift 2 ;;
    --ix) IX="${2:-}"; shift 2 ;;
    --arch) ARCH="${2:-}"; shift 2 ;;
    --norm-type) NORM_TYPE="${2:-}"; shift 2 ;;
    --python-bin) PYTHON_BIN="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERR] Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" || -z "$DATA_ROOT" || -z "$MODEL" ]]; then
  echo "[ERR] Missing required arguments."
  usage
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERR] Checkpoint not found: $CHECKPOINT"
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERR] Data root not found: $DATA_ROOT"
  exit 1
fi

DATA_ROOT_ABS="$(cd "$DATA_ROOT" && pwd)"
CHECKPOINT_ABS="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
WEIGHTS_DIR_ABS="$(mkdir -p "$WEIGHTS_DIR" && cd "$WEIGHTS_DIR" && pwd)"

detect_patch_size() {
  local inferred=""
  local vals=()
  local cfgs=()

  # 1) Prefer existing config for the same model if present.
  if [[ -f "$WEIGHTS_DIR_ABS/$MODEL/config.yaml" ]]; then
    inferred="$(awk -F':' '/^patch_size:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$WEIGHTS_DIR_ABS/$MODEL/config.yaml" || true)"
    if [[ -n "$inferred" ]]; then
      echo "$inferred"
      return 0
    fi
  fi

  # 2) Otherwise infer from all existing model configs if all are identical.
  while IFS= read -r cfg; do
    cfgs+=("$cfg")
  done < <(find "$WEIGHTS_DIR_ABS" -mindepth 2 -maxdepth 2 -type f -name config.yaml | sort)

  if [[ "${#cfgs[@]}" -eq 0 ]]; then
    return 0
  fi

  for cfg in "${cfgs[@]}"; do
    val="$(awk -F':' '/^patch_size:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$cfg" || true)"
    [[ -n "$val" ]] && vals+=("$val")
  done

  if [[ "${#vals[@]}" -eq 0 ]]; then
    return 0
  fi

  uniq_vals="$(printf "%s\n" "${vals[@]}" | sort -u)"
  uniq_count="$(printf "%s\n" "$uniq_vals" | sed '/^$/d' | wc -l | tr -d ' ')"

  if [[ "$uniq_count" == "1" ]]; then
    printf "%s\n" "$uniq_vals" | sed '/^$/d' | head -n 1
    return 0
  fi

  return 0
}

if [[ -z "$PATCH_SIZE" ]]; then
  PATCH_SIZE="$(detect_patch_size || true)"
  if [[ -n "$PATCH_SIZE" ]]; then
    echo "[INFO] Auto-detected patch_size=$PATCH_SIZE from existing config.yaml"
  else
    PATCH_SIZE="256"
    echo "[WARN] Could not auto-detect patch_size. Using default patch_size=256"
  fi
fi

if [[ -z "$CSV_DIR" ]]; then
  data_parent="$(dirname "$(dirname "$DATA_ROOT_ABS")")"
  CSV_DIR="$data_parent/csv_${MODEL}/$(basename "$DATA_ROOT_ABS")"
fi
if [[ -z "$OUT_DIR" ]]; then
  if [[ -d /workspace || ! -e /workspace ]]; then
    mkdir -p /workspace/output
    OUT_DIR="/workspace/output/robust_eval_${MODEL}/$(basename "$DATA_ROOT_ABS")"
  else
    OUT_DIR="$SCRIPT_DIR/output/robust_eval_${MODEL}/$(basename "$DATA_ROOT_ABS")"
  fi
fi

CSV_DIR_ABS="$(mkdir -p "$CSV_DIR" && cd "$CSV_DIR" && pwd)"
OUT_DIR_ABS="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

IFS=',' read -r -a DOMAINS <<< "$DOMAINS_CSV"
if [[ "${#DOMAINS[@]}" -eq 0 ]]; then
  echo "[ERR] Empty domains list."
  exit 1
fi

MODEL_DIR="$WEIGHTS_DIR_ABS/$MODEL"
mkdir -p "$MODEL_DIR"
ln -sfn "$CHECKPOINT_ABS" "$MODEL_DIR/weights.pth"

if [[ "$PATCH_SIZE" == "null" ]]; then
  patch_yaml="null"
else
  patch_yaml="$PATCH_SIZE"
fi

cat > "$MODEL_DIR/config.yaml" <<EOF
arch: $ARCH
model_name: $MODEL
norm_type: $NORM_TYPE
patch_size: $patch_yaml
weights_file: weights.pth
EOF

echo "[INFO] Model registered: $MODEL_DIR"
echo "[INFO] Data root: $DATA_ROOT_ABS"
echo "[INFO] CSV dir: $CSV_DIR_ABS"
echo "[INFO] Output dir: $OUT_DIR_ABS"

for d in "${DOMAINS[@]}"; do
  src="$DATA_ROOT_ABS/$d"
  if [[ ! -d "$src" ]]; then
    echo "[ERR] Missing domain folder: $src"
    exit 1
  fi

  if [[ "$MAX_FILES" == "all" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/create_csv.py" \
      "$src" \
      "$CSV_DIR_ABS/$d.csv" \
      --dir "$d" \
      --max-files 999999999
  else
    "$PYTHON_BIN" "$SCRIPT_DIR/create_csv.py" \
      "$src" \
      "$CSV_DIR_ABS/$d.csv" \
      --dir "$d" \
      --max-files "$MAX_FILES"
  fi

  rows=$(( $(wc -l < "$CSV_DIR_ABS/$d.csv") - 1 ))
  echo "[CSV] $d: $rows rows"
  if (( rows <= 0 )); then
    echo "[ERR] Empty CSV generated for domain: $d"
    exit 1
  fi
done

for d in "${DOMAINS[@]}"; do
  in_csv="$CSV_DIR_ABS/$d.csv"
  out_csv="$OUT_DIR_ABS/${d}_scored.csv"

  echo "[TEST] $d"
  if [[ -n "$CUDA_VISIBLE_IDS" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_IDS" "$PYTHON_BIN" "$SCRIPT_DIR/main.py" \
      --in_csv "$in_csv" \
      --out_csv "$out_csv" \
      --weights_dir "$WEIGHTS_DIR_ABS" \
      --models "$MODEL" \
      --device "$DEVICE"
  else
    "$PYTHON_BIN" "$SCRIPT_DIR/main.py" \
      --in_csv "$in_csv" \
      --out_csv "$out_csv" \
      --weights_dir "$WEIGHTS_DIR_ABS" \
      --models "$MODEL" \
      --device "$DEVICE"
  fi
done

REAL="$OUT_DIR_ABS/real_scored.csv"
METRIC="$OUT_DIR_ABS/metrics_summary.csv"

if [[ ! -f "$REAL" ]]; then
  echo "[ERR] real_scored.csv not found: $REAL"
  exit 1
fi

rm -f "$METRIC"
shopt -s nullglob
scored_files=("$OUT_DIR_ABS"/*_scored.csv)
shopt -u nullglob

if [[ "${#scored_files[@]}" -eq 0 ]]; then
  echo "[ERR] No scored files found in: $OUT_DIR_ABS"
  exit 1
fi

eval_count=0
for fake in "${scored_files[@]}"; do
  [[ "$fake" == "$REAL" ]] && continue
  name="$(basename "$fake" _scored.csv)"
  echo "[EVAL] $name"
  "$PYTHON_BIN" "$SCRIPT_DIR/eval.py" \
    --real "$REAL" \
    --fake "$fake" \
    --ix "$IX" \
    --out-csv "$METRIC"
  eval_count=$((eval_count + 1))
done

if (( eval_count == 0 )); then
  echo "[ERR] No fake scored CSV found for evaluation."
  exit 1
fi

echo "[DONE] metrics saved: $METRIC"
