#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"

export Master='spark://spark-master:7077'
export PYTHONIOENCODING=utf8
#export Master='spark://172.17.0.2:7077'
function execut {
/opt/spark-3.0.0-bin-hadoop3.2/bin/spark-submit  \
     --master $Master \
     --total-executor-cores 4 \
     --executor-memory 16G \
     --executor-cores 2 \
     --conf spark.plugins=com.nvidia.spark.SQLPlugin \
     --conf spark.executor.resource.gpu.amount=1 \
     $currentPath/../../src/preprocess/BX/spark.py  \
     --input_dir $currentPath/../../../data/BX-CSV-Dump/raw \
     --output_dir $currentPath/../../../data/BX-CSV-Dump/preprocess | tee ./submit_gpu.log
}


starttime=`date +'%Y-%m-%d %H:%M:%S'`
execut
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo '==================================================='
echo "the job execute time： "$((end_seconds-start_seconds))"s"
echo '==================================================='