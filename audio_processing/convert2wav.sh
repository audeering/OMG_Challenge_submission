#!/bin/bash
for d in test_data/**/*.mp4; do  
     if ! [ -f "$d".wav ];then     
      echo "$d"
      ffmpeg -y -i "$d" "$d".wav
     fi
done
