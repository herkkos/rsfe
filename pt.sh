#!/bin/bash -l
#SBATCH -J puheentunnistusHerkko
#SBATCH -o /wrk/salone35/logs/output_%j.txt
#SBATCH -e /wrk/salone35/logs/errors_%j.txt
#SBATCH -t 03:00:00
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH -p serial
#SBATCH --mail-type=ALL
#SBATCH --mail-user=herkko.salonen@tuni.fi
#SBATCH --mem-per-cpu=2000

WEV=/wrk/salone35/wev-$SLURM_JOB_ID
mkdir $WEV ; cd $WEV

module load aaltoasr
echo "Puheentunnistus"

for file in /wrk/salone35/denoised/*
do
        identifier=${file:29:4}

        echo $identifier

        if [ $identifier == "bike" ]
        then
                aaltoasr-align -n1 -o "${file:23:5}output.txt" -t /wrk/salone35/bike.txt -T "${file:23:5}output.TextGrid" $file
        elif [ $identifier == "carr" ]
        then
                aaltoasr-align -n1 -o "${file:23:5}output.txt" -t /wrk/salone35/carrot.txt -T "${file:23:5}output.TextGrid" $file
        elif [ $identifier == "pake" ]
        then
                aaltoasr-align -n1 -o "${file:23:5}output.txt" -t /wrk/salone35/paketti.txt -T "${file:23:5}output.TextGrid" $file
        else
       		echo "fail"
        fi
done
