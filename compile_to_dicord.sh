#!/bin/sh

CI_COMMIT_SHA="$1"
DISCORD_WEBHOOK="$2"


echo mkdir out
# mkdir out

for f in $(git diff-tree --no-commit-id --name-status -r $CI_COMMIT_SHA)
do
    f="$(echo "$f" | sed 's/"//g')"
    case $f in
        *.py)
            echo Changed python file $f
            ;;
        *)
            echo "Changes in " $f
    esac
done
# cd out || exit 1
# for o in *
# do
#     case "$o" in
#         *.html)
#             name="$(echo "$o" | sed "s/\.html//")"
#             # echo wkhtmltopdf "$name".html "$name".pdf
#             # wkhtmltopdf "$name".html "$name".pdf
#             curl --form file=@"$name".html "$DISCORD_WEBHOOK"
#             # curl --form file=@"$name".pdf  "$DISCORD_WEBHOOK"
#             ;;
#     esac
# done
echo "done deer"