export dss=(10 20 50 75 100 125 150 175 200 250 300 350 400 450 500)

# Function to process a chunk of runs
process_chunk() {
  local chunk_id=$1
  local start=$2
  local end=$3
  echo "Scheduling parallel chunk $chunk_id (seeds $start to $end)"

  mkdir -p csv_parallel/${chunk_id}/none
  for ds in ${dss[@]}; do
    for seed in $(seq $start $end); do
      python main.py --epochs $ds --batch_size 50 --gamma 0.9 --seed $seed --lambda_ratio 0.0 --output_dir csv_parallel/${chunk_id}/none > /dev/null 2>&1
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

mkdir -p csv/none
for ds in ${dss[@]}; do
  for ((i = 0; i < PARALLEL_JOBS; i++)); do
    partial_file_path="csv_parallel/${i}/none/epochs=$ds,lambda_ratio=*.csv"
    partial_file_name=$(basename $partial_file_path)
    cat $partial_file_path >> "csv/none/$partial_file_name"
  done
done
