#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
starttime=`date +'%Y-%m-%d %H:%M:%S'`

#bash $currentPath/submit_subset_deepfm.sh
#bash $currentPath/submit_subset_xdeepfm.sh
#bash $currentPath/submit_subset_deepcrossing.sh
#bash $currentPath/submit_subset_dcn.sh
#bash $currentPath/submit_subset_nfm.sh
#bash $currentPath/submit_subset_afm.sh
#bash $currentPath/submit_subset_fibinet.sh
#bash $currentPath/submit_subset_pnn.sh
#bash $currentPath/submit_subset_fgcnn.sh
bash $currentPath/submit_subset_ccpm.sh

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo '==================================================='
echo "the job execute time： "$((end_seconds-start_seconds))"s"
echo '==================================================='