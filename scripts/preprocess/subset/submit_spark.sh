#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"

cd ../../../
pwdPath="$(pwd)"
export PYSPARK_PYTHON=/opt/conda/bin/python3
export Master='spark://spark-master:7077'
export PYTHONIOENCODING=utf8

outptut=$pwdPath/data/kaggle/subset/process
if [ -d $outptut ]; then
    echo "remove directory $outptut"
    rm -rf $outptut
fi

#export Master='spark://172.17.0.2:7077'
function execut {
/opt/spark-3.0.0-bin-hadoop3.2/bin/spark-submit  \
     --master $Master \
     --total-executor-cores 4 \
     --executor-memory 16G \
     --executor-cores 2 \
     --conf spark.plugins=com.nvidia.spark.SQLPlugin \
     --conf spark.executor.resource.gpu.amount=1 \
     $pwdPath/src/preprocess/subset/spark.py  \
     --input_path $pwdPath/data/kaggle/subset/raw/train_subset_1000000.csv \
     --output_path $outptut | tee ./submit_gpu.log
}


starttime=`date +'%Y-%m-%d %H:%M:%S'`
execut
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo '==================================================='
echo "the job execute time： "$((end_seconds-start_seconds))"s"
echo '==================================================='