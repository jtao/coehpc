#!/bin/sh
for i in `seq 56`
do
  echo "./t.R $i" >> commands.in
done
