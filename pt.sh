#!/bin/bash -l
#SBATCH -J puheentunnistus
#SBATCH -o /wrk/*USER*/logs/output_%j.txt
#SBATCH -e /wrk/*USER*/logs/errors_%j.txt
#SBATCH -t 03:00:00
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH -p serial
#SBATCH --mail-type=ALL
#SBATCH --mail-user=*EMAIL*
#SBATCH --mem-per-cpu=2000

WEV=/wrk/*USER*/wev-$SLURM_JOB_ID
mkdir $WEV ; cd $WEV

module load aaltoasr
echo "Puheentunnistus"

for file in /wrk/*USER*/denoised/*
do
        identifier=${file:29:4}

        echo $identifier

        if [ $identifier == "bike" ]
        then
                aaltoasr-align -n1 -o "${file:23:5}output.txt" -t /wrk/*USER*/bike.txt -T "${file:23:5}output.TextGrid" $file
        elif [ $identifier == "carr" ]
        then
                aaltoasr-align -n1 -o "${file:23:5}output.txt" -t /wrk/*USER*/carrot.txt -T "${file:23:5}output.TextGrid" $file
        elif [ $identifier == "pake" ]
        then
                aaltoasr-align -n1 -o "${file:23:5}output.txt" -t /wrk/*USER*/paketti.txt -T "${file:23:5}output.TextGrid" $file
        else
       		echo "fail"
        fi
done
