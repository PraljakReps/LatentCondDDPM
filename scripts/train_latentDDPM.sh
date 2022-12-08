#!/usr/bin/env sh


python -V
export DIR="$(dirname "$(pwd)")"
source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}


export SEED=42
export batch_size=32
export learning_rate=1e-3
export epochs=10

# model hyperparameters
export img_size=32
export c_in=1
export c_out=1
export first_num_channel=64
export time_dim=256
export num_layers=3
export bn_layers=2
export rep_learning=True
export alpha=0.99
export beta=10
export z_dim=5

# filepaths
export fig_output_path='.././output/latentDDPM/mnist_samples_'
export output_model_path='.././output/latentDDPM/model_pretrained_weights'
export output_csv_path='.././output/latentDDPM/model_train_history.csv'

python ../train_latentDDPM.py \
	--SEED ${SEED} \
	--batch_size ${batch_size} \
	--learning_rate ${learning_rate} \
	--epochs ${epochs} \
	--img_size ${img_size} \
	--c_in ${c_in} \
	--c_out ${c_out} \
	--first_num_channel ${first_num_channel} \
	--time_dim ${time_dim} \
	--num_layers ${num_layers} \
	--bn_layers ${bn_layers} \
	--rep_learning ${rep_learning} \
	--alpha ${alpha} \
	--beta ${beta} \
	--z_dim ${z_dim} \
	--fig_output_path ${fig_output_path} \
	--output_model_path ${output_model_path} \
	--output_csv_path ${output_csv_path} \
