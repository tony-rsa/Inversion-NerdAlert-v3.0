#!/bin/bash
container_url="https://inversionrecruitment.blob.core.windows.net/find-the-code"

rm -rf images/find-the-code/png-tiles
mkdir -p images/find-the-code/png-tiles
pushd images/find-the-code/png-tiles

for image_index in {1..1200}; do
    clear 

echo -e "Now downloading /033[34m$image_index/033[0m of 1200.."
  sleep 3 &&  curl -X GET -H "x-ms-date: $(date -u)" -o "($image_index).png" "$container_url/($image_index).png?ss=bfqt&srt=sco"
done

find -type f -name *.png "`pwd`/images/find-the-code/png-tiles" > image-tiles.txt

cct=`wc -l image-tiles.txt | awk "{printf $1}"`

clear 
sleep 1
echo -e "/033[32m$cct/033[0m image tiles where download successfully."
