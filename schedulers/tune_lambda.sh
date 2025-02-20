TOTAL_RUNS=30
PARALLEL_JOBS=30

dss=(50 200 400)

# Function to process a chunk of runs
process_chunk() {
  local chunk_id=$1
  local start=$2
  local end=$3
  echo "Scheduling parallel chunk $chunk_id (seeds $start to $end)"

  mkdir -p csv_parallel/${chunk_id}/lambda_search

  for ds in ${dss[@]}; do
    for lambda_ratio in `seq -f "%.1f" 0.0 0.1 0.9`; do
      for seed in $(seq $start $end); do
        python main.py --epochs $ds --batch_size 50 --gamma 0.9 --seed $seed --lambda_ratio $lambda_ratio --output_dir csv_parallel/${chunk_id}/lambda_search > /dev/null 2>&1
      done
    done
  done
}

# Divide runs into chunks and parallelize the chunks
CHUNK_SIZE=$((TOTAL_RUNS / PARALLEL_JOBS))
for ((i = 0; i < PARALLEL_JOBS; i++)); do
  start=$((i * CHUNK_SIZE + 1))
  end=$(((i + 1) * CHUNK_SIZE))
  if [ $i -eq $((PARALLEL_JOBS - 1)) ]; then
    # Adjust the end for the last chunk to ensure all runs are processed
    end=$TOTAL_RUNS
  fi

  process_chunk $i $start $end &
done
wait

mkdir -p csv/lambda_search
for ds in ${dss[@]}; do
  for lambda_ratio in `seq -f "%.1f" 0.0 0.1 0.9`; do
    for ((i = 0; i < PARALLEL_JOBS; i++)); do
      cat csv_parallel/${i}/lambda_search/epochs=$ds,lambda_ratio=$lambda_ratio.csv >> csv/lambda_search/epochs=$ds,lambda_ratio=$lambda_ratio.csv
    done
  done
done
