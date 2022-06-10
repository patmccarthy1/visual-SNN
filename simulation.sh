#PBS -lwalltime=48:00:00
#PBS -lselect=1:ncpus=8:mem=96gb
module load anaconda3/personal
mkdir $WORK/$PBS_JOBID
mkdir input_data
mkdir output_data
mv $HOME/input_data/* input_data
python3 $HOME/simulation.py 
mv * $WORK/$PBS_JOBID

