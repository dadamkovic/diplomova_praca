#!/bin/bash

print_help () {
	echo "Will run multiple training sessions one after the other."
	echo "Syntax:"
	echo "./train batch <num of training runs> <num of frames>"
}

if [ $# -eq 0 ]
then
	print_help
	exit
fi

while getopts ":h" option
do
	case $option in
		h)
			print_help
			exit;;
	esac
done

num_runs=$1
num_frames=$2

for (( i = 0; i < num_runs; i++ ))
do
	r0=$(echo "$((5000+RANDOM%(15000-5000))).$((RANDOM%999))")
	r1=$(echo "$((0)).$((RANDOM%999))")
	r2=$(echo "$((5+RANDOM%(100-5))).$((RANDOM%999))")
	r3=$(echo "$((5+RANDOM%(100-5))).$((RANDOM%999))")
	echo "python agent_minitaur_inspired --mode=train --frames=$num_frames --rew_params $r0 $r1 $r2 $r3"
	python agent_minitaur_inspired.py --mode=train --frames=$num_frames -rew_params $r0 $r1 $r2 $r3
done
