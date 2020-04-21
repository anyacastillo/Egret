#! /bin/bash
for ((i=0;i<=43;i++))
do
	python test_approximations.py $i &> console_$i.out &
done
