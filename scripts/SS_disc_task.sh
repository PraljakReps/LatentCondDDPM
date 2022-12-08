#!/usr/bin/env sh


python -V
export DIR="$(dirname "$(pwd)")"
source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}


export SEED=42
export batch_size=32
export learning_rate=1e-3
export epochs=10
export perc_size='0.01,0.05,0.1,0.3,0.5,1.0'

# model hyperparameters
export img_size=32
export c_in=1
export c_out=1
export first_num_channel=64
export time_dim=256
export num_layers=3
export bn_layers=2
export disc_num_layers=2
export disc_width=100
export rep_learning=True
export xi=1
export alpha=0.99
export beta=10
export gamma=10
export z_dim=10


# filepaths
export fig_output_path='.././output/SS_disc_exp/mnist_samples_'
export output_model_path='.././output/SS_disc_exp/model_pretrained_weights'
export output_csv_path='.././output/SS_disc_exp/model_train_history.csv'
export tsne_path='.././output/SS_disc_exp/tsne_plot_'

python ../SS_disc_task.py \
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
	--disc_num_layers ${disc_num_layers} \
	--disc_width ${disc_width} \
	--rep_learning ${rep_learning} \
	--xi ${xi} \
	--alpha ${alpha} \
	--beta ${beta} \
	--gamma ${gamma} \
	--z_dim ${z_dim} \
	--fig_output_path ${fig_output_path} \
	--output_model_path ${output_model_path} \
	--output_csv_path ${output_csv_path} \
	--tsne_path ${tsne_path} \
	--perc_size ${perc_size} \
