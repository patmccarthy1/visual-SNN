#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=8:mem=96gb
module load anaconda3/personal
conda install brian2
conda install -c anaconda opencv 
conda install matplotlib
mkdir $WORK/$PBS_JOBID
mkdir output_data
mv $HOME/input_data $WORK/$PBS_JOBID
python $HOME/simulation.py 
mv * $WORK/$PBS_JOBID

