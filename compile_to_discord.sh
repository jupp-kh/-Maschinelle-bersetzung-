#!/bin/bash

CI_COMMIT_SHA="$1"
## discord webhook
DISCORD_WEBHOOK="$2"


data_msg="Report: All changes in repository \nDate: $(date)\n"


output=($(git diff-tree --no-commit-id --name-status -r $CI_COMMIT_SHA))
LEN=${#output[*]}


for ((i=0;i<LEN;i++));
do
    x=$i%2
    if [[ $x -eq 0 ]];
    then 
        f=${output[$i]}
        f="$(echo "$f" | sed 's/"//g' | sed 's/ //g')"
        # echo $f
        i=$i+1
        case $f in
            A*)
                data_msg+="- Added new file ${output[$i]} \n"
                ;;
            M*)
                data_msg+="- Modified file: ${output[$i]} \n"
        esac
    fi
done

curl -H "Content-Type: application/json" -X POST -d "{\"content\": \"${data_msg}\"}" $DISCORD_WEBHOOK

echo "done deer"