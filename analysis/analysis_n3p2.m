%% read in data

% spike data
L0_spikes = readmatrix('../output_data/simulation_5/layer_0_full_spikes.csv');
L0_spikes_train_idx = find(L0_spikes(2,:)>8,1);
L0_spikes_train = L0_spikes(:,1:L0_spikes_train_idx);
L0_spikes_test = L0_spikes(:,L0_spikes_train_idx:end);
L1_exc_spikes = readmatrix('../output_data/simulation_5/layer_1_excitatory_full_spikes.csv');
L1_exc_spikes_train_idx = find(L1_exc_spikes(2,:)>8,1);
L1_exc_spikes_train = L1_exc_spikes(:,1:L1_exc_spikes_train_idx);
L1_exc_spikes_test = L1_exc_spikes(:,L1_exc_spikes_train_idx:end);
% L1_inh_spikes = readmatrix('../output_data/simulation_5/layer_1_inhibitory_full_spikes.csv');
% L1_inh_spikes_train_idx = find(L1_inh_spikes(2,:)>8,1);
% L1_inh_spikes_train = L1_inh_spikes(:,1:L1_inh_spikes_train_idx);
% L1_inh_spikes_test = L1_inh_spikes(:,L1_inh_spikes_train_idx:end);
L2_exc_spikes = readmatrix('../output_data/simulation_5/layer_2_excitatory_full_spikes.csv');
L2_exc_spikes_train_idx = find(L2_exc_spikes(2,:)>8,1);
L2_exc_spikes_train = L2_exc_spikes(:,1:L2_exc_spikes_train_idx);
L2_exc_spikes_test = L2_exc_spikes(:,L2_exc_spikes_train_idx:end);
L2_inh_spikes = readmatrix('../output_data/simulation_5/layer_2_inhibitory_full_spikes.csv');
L2_inh_spikes_train_idx = find(L2_inh_spikes(2,:)>8,1);
L2_inh_spikes_train = L2_inh_spikes(:,1:L2_inh_spikes_train_idx);
L2_inh_spikes_test = L2_inh_spikes(:,L2_inh_spikes_train_idx:end);
L3_exc_spikes = readmatrix('../output_data/simulation_5/layer_3_excitatory_full_spikes.csv');
L3_exc_spikes_train_idx = find(L3_exc_spikes(2,:)>8,1);
L3_exc_spikes_train = L3_exc_spikes(:,1:L3_exc_spikes_train_idx);
L3_exc_spikes_test = L3_exc_spikes(:,L3_exc_spikes_train_idx:end);
L3_inh_spikes = readmatrix('../output_data/simulation_5/layer_3_inhibitory_full_spikes.csv');
L3_inh_spikes_train_idx = find(L3_inh_spikes(2,:)>8,1);
L3_inh_spikes_train = L3_inh_spikes(:,1:L3_inh_spikes_train_idx);
L3_inh_spikes_test = L3_inh_spikes(:,L3_inh_spikes_train_idx:end);
L4_exc_spikes = readmatrix('../output_data/simulation_5/layer_4_excitatory_full_spikes.csv');
L4_exc_spikes_train_idx = find(L4_exc_spikes(2,:)>8,1);
L4_exc_spikes_train = L4_exc_spikes(:,1:L4_exc_spikes_train_idx);
L4_exc_spikes_test = L4_exc_spikes(:,L4_exc_spikes_train_idx:end);
L4_inh_spikes = readmatrix('../output_data/simulation_5/layer_4_inhibitory_full_spikes.csv');
L4_inh_spikes_train_idx = find(L4_inh_spikes(2,:)>8,1);
L4_inh_spikes_train = L4_inh_spikes(:,1:L4_inh_spikes_train_idx);
L4_inh_spikes_test = L4_inh_spikes(:,L4_inh_spikes_train_idx:end);

% weight data
layer_0_layer_1_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_0s.csv');
layer_0_layer_1_exc_weights_0_2s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im0_0_2s.csv');
layer_0_layer_1_exc_weights_0_4s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im0_0_4s.csv');
layer_0_layer_1_exc_weights_0_6s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im0_0_6000000000000001s.csv');
layer_0_layer_1_exc_weights_0_8s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im0_0_8s.csv');
layer_0_layer_1_exc_weights_1_0s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im0_1_0s.csv');
layer_0_layer_1_exc_weights_1_2s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im1_0_2s.csv');
layer_0_layer_1_exc_weights_1_4s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im1_0_4s.csv');
layer_0_layer_1_exc_weights_1_6s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im1_0_6000000000000001s.csv');
layer_0_layer_1_exc_weights_1_8s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im1_0_8s.csv');
layer_0_layer_1_exc_weights_2_0s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im1_1_0s.csv');
layer_0_layer_1_exc_weights_2_2s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im2_0_2s.csv');
layer_0_layer_1_exc_weights_2_4s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im2_0_4s.csv');
layer_0_layer_1_exc_weights_2_6s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im2_0_6000000000000001s.csv');
layer_0_layer_1_exc_weights_2_8s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im2_0_8s.csv');
layer_0_layer_1_exc_weights_3_0s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im2_1_0s.csv');
layer_0_layer_1_exc_weights_3_2s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im3_0_2s.csv');
layer_0_layer_1_exc_weights_3_4s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im3_0_4s.csv');
layer_0_layer_1_exc_weights_3_6s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im3_0_6000000000000001s.csv');
layer_0_layer_1_exc_weights_3_8s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im3_0_8s.csv');
layer_0_layer_1_exc_weights_4_0s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im3_1_0s.csv');
layer_0_layer_1_exc_weights_4_2s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im4_0_2s.csv');
layer_0_layer_1_exc_weights_4_4s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im4_0_4s.csv');
layer_0_layer_1_exc_weights_4_6s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im4_0_6000000000000001s.csv');
layer_0_layer_1_exc_weights_4_8s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im4_0_8s.csv');
layer_0_layer_1_exc_weights_5_0s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im4_1_0s.csv');
layer_0_layer_1_exc_weights_5_2s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im5_0_2s.csv');
layer_0_layer_1_exc_weights_5_4s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im5_0_4s.csv');
layer_0_layer_1_exc_weights_5_6s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im5_0_6000000000000001s.csv');
layer_0_layer_1_exc_weights_5_8s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im5_0_8s.csv');
layer_0_layer_1_exc_weights_6_0s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im5_1_0s.csv');
layer_0_layer_1_exc_weights_6_2s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im6_0_2s.csv');
layer_0_layer_1_exc_weights_6_4s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im6_0_4s.csv');
layer_0_layer_1_exc_weights_6_6s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im6_0_6000000000000001s.csv');
layer_0_layer_1_exc_weights_6_8s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im6_0_8s.csv');
layer_0_layer_1_exc_weights_7_0s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im6_1_0s.csv');
layer_0_layer_1_exc_weights_7_2s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im7_0_2s.csv');
layer_0_layer_1_exc_weights_7_4s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im7_0_4s.csv');
layer_0_layer_1_exc_weights_7_6s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im7_0_6000000000000001s.csv');
layer_0_layer_1_exc_weights_7_8s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im7_0_8s.csv');
layer_0_layer_1_exc_weights_8_0s = readmatrix('../output_data/simulation_5/layer_0_layer_1_exc_weights_im7_1_0s.csv');
 
layer_1_exc_layer_2_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_0s.csv');
layer_1_exc_layer_2_exc_weights_0_2s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im0_0_2s.csv');
layer_1_exc_layer_2_exc_weights_0_4s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im0_0_4s.csv');
layer_1_exc_layer_2_exc_weights_0_6s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im0_0_6000000000000001s.csv');
layer_1_exc_layer_2_exc_weights_0_8s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im0_0_8s.csv');
layer_1_exc_layer_2_exc_weights_1_0s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im0_1_0s.csv');
layer_1_exc_layer_2_exc_weights_1_2s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im1_0_2s.csv');
layer_1_exc_layer_2_exc_weights_1_4s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im1_0_4s.csv');
layer_1_exc_layer_2_exc_weights_1_6s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im1_0_6000000000000001s.csv');
layer_1_exc_layer_2_exc_weights_1_8s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im1_0_8s.csv');
layer_1_exc_layer_2_exc_weights_2_0s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im1_1_0s.csv');
layer_1_exc_layer_2_exc_weights_2_2s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im2_0_2s.csv');
layer_1_exc_layer_2_exc_weights_2_4s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im2_0_4s.csv');
layer_1_exc_layer_2_exc_weights_2_6s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im2_0_6000000000000001s.csv');
layer_1_exc_layer_2_exc_weights_2_8s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im2_0_8s.csv');
layer_1_exc_layer_2_exc_weights_3_0s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im2_1_0s.csv');
layer_1_exc_layer_2_exc_weights_3_2s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im3_0_2s.csv');
layer_1_exc_layer_2_exc_weights_3_4s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im3_0_4s.csv');
layer_1_exc_layer_2_exc_weights_3_6s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im3_0_6000000000000001s.csv');
layer_1_exc_layer_2_exc_weights_3_8s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im3_0_8s.csv');
layer_1_exc_layer_2_exc_weights_4_0s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im3_1_0s.csv');
layer_1_exc_layer_2_exc_weights_4_2s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im4_0_2s.csv');
layer_1_exc_layer_2_exc_weights_4_4s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im4_0_4s.csv');
layer_1_exc_layer_2_exc_weights_4_6s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im4_0_6000000000000001s.csv');
layer_1_exc_layer_2_exc_weights_4_8s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im4_0_8s.csv');
layer_1_exc_layer_2_exc_weights_5_0s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im4_1_0s.csv');
layer_1_exc_layer_2_exc_weights_5_2s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im5_0_2s.csv');
layer_1_exc_layer_2_exc_weights_5_4s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im5_0_4s.csv');
layer_1_exc_layer_2_exc_weights_5_6s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im5_0_6000000000000001s.csv');
layer_1_exc_layer_2_exc_weights_5_8s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im5_0_8s.csv');
layer_1_exc_layer_2_exc_weights_6_0s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im5_1_0s.csv');
layer_1_exc_layer_2_exc_weights_6_2s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im6_0_2s.csv');
layer_1_exc_layer_2_exc_weights_6_4s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im6_0_4s.csv');
layer_1_exc_layer_2_exc_weights_6_6s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im6_0_6000000000000001s.csv');
layer_1_exc_layer_2_exc_weights_6_8s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im6_0_8s.csv');
layer_1_exc_layer_2_exc_weights_7_0s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im6_1_0s.csv');
layer_1_exc_layer_2_exc_weights_7_2s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im7_0_2s.csv');
layer_1_exc_layer_2_exc_weights_7_4s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im7_0_4s.csv');
layer_1_exc_layer_2_exc_weights_7_6s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im7_0_6000000000000001s.csv');
layer_1_exc_layer_2_exc_weights_7_8s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im7_0_8s.csv');
layer_1_exc_layer_2_exc_weights_8_0s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_2_exc_weights_im7_1_0s.csv');
 
layer_2_exc_layer_3_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_0s.csv');
layer_2_exc_layer_3_exc_weights_0_2s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im0_0_2s.csv');
layer_2_exc_layer_3_exc_weights_0_4s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im0_0_4s.csv');
layer_2_exc_layer_3_exc_weights_0_6s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im0_0_6000000000000001s.csv');
layer_2_exc_layer_3_exc_weights_0_8s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im0_0_8s.csv');
layer_2_exc_layer_3_exc_weights_1_0s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im0_1_0s.csv');
layer_2_exc_layer_3_exc_weights_1_2s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im1_0_2s.csv');
layer_2_exc_layer_3_exc_weights_1_4s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im1_0_4s.csv');
layer_2_exc_layer_3_exc_weights_1_6s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im1_0_6000000000000001s.csv');
layer_2_exc_layer_3_exc_weights_1_8s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im1_0_8s.csv');
layer_2_exc_layer_3_exc_weights_2_0s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im1_1_0s.csv');
layer_2_exc_layer_3_exc_weights_2_2s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im2_0_2s.csv');
layer_2_exc_layer_3_exc_weights_2_4s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im2_0_4s.csv');
layer_2_exc_layer_3_exc_weights_2_6s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im2_0_6000000000000001s.csv');
layer_2_exc_layer_3_exc_weights_2_8s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im2_0_8s.csv');
layer_2_exc_layer_3_exc_weights_3_0s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im2_1_0s.csv');
layer_2_exc_layer_3_exc_weights_3_2s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im3_0_2s.csv');
layer_2_exc_layer_3_exc_weights_3_4s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im3_0_4s.csv');
layer_2_exc_layer_3_exc_weights_3_6s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im3_0_6000000000000001s.csv');
layer_2_exc_layer_3_exc_weights_3_8s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im3_0_8s.csv');
layer_2_exc_layer_3_exc_weights_4_0s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im3_1_0s.csv');
layer_2_exc_layer_3_exc_weights_4_2s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im4_0_2s.csv');
layer_2_exc_layer_3_exc_weights_4_4s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im4_0_4s.csv');
layer_2_exc_layer_3_exc_weights_4_6s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im4_0_6000000000000001s.csv');
layer_2_exc_layer_3_exc_weights_4_8s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im4_0_8s.csv');
layer_2_exc_layer_3_exc_weights_5_0s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im4_1_0s.csv');
layer_2_exc_layer_3_exc_weights_5_2s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im5_0_2s.csv');
layer_2_exc_layer_3_exc_weights_5_4s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im5_0_4s.csv');
layer_2_exc_layer_3_exc_weights_5_6s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im5_0_6000000000000001s.csv');
layer_2_exc_layer_3_exc_weights_5_8s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im5_0_8s.csv');
layer_2_exc_layer_3_exc_weights_6_0s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im5_1_0s.csv');
layer_2_exc_layer_3_exc_weights_6_2s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im6_0_2s.csv');
layer_2_exc_layer_3_exc_weights_6_4s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im6_0_4s.csv');
layer_2_exc_layer_3_exc_weights_6_6s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im6_0_6000000000000001s.csv');
layer_2_exc_layer_3_exc_weights_6_8s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im6_0_8s.csv');
layer_2_exc_layer_3_exc_weights_7_0s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im6_1_0s.csv');
layer_2_exc_layer_3_exc_weights_7_2s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im7_0_2s.csv');
layer_2_exc_layer_3_exc_weights_7_4s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im7_0_4s.csv');
layer_2_exc_layer_3_exc_weights_7_6s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im7_0_6000000000000001s.csv');
layer_2_exc_layer_3_exc_weights_7_8s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im7_0_8s.csv');
layer_2_exc_layer_3_exc_weights_8_0s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_3_exc_weights_im7_1_0s.csv');
 
layer_3_exc_layer_4_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_0s.csv');
layer_3_exc_layer_4_exc_weights_0_2s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im0_0_2s.csv');
layer_3_exc_layer_4_exc_weights_0_4s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im0_0_4s.csv');
layer_3_exc_layer_4_exc_weights_0_6s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im0_0_6000000000000001s.csv');
layer_3_exc_layer_4_exc_weights_0_8s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im0_0_8s.csv');
layer_3_exc_layer_4_exc_weights_1_0s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im0_1_0s.csv');
layer_3_exc_layer_4_exc_weights_1_2s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im1_0_2s.csv');
layer_3_exc_layer_4_exc_weights_1_4s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im1_0_4s.csv');
layer_3_exc_layer_4_exc_weights_1_6s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im1_0_6000000000000001s.csv');
layer_3_exc_layer_4_exc_weights_1_8s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im1_0_8s.csv');
layer_3_exc_layer_4_exc_weights_2_0s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im1_1_0s.csv');
layer_3_exc_layer_4_exc_weights_2_2s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im2_0_2s.csv');
layer_3_exc_layer_4_exc_weights_2_4s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im2_0_4s.csv');
layer_3_exc_layer_4_exc_weights_2_6s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im2_0_6000000000000001s.csv');
layer_3_exc_layer_4_exc_weights_2_8s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im2_0_8s.csv');
layer_3_exc_layer_4_exc_weights_3_0s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im2_1_0s.csv');
layer_3_exc_layer_4_exc_weights_3_2s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im3_0_2s.csv');
layer_3_exc_layer_4_exc_weights_3_4s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im3_0_4s.csv');
layer_3_exc_layer_4_exc_weights_3_6s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im3_0_6000000000000001s.csv');
layer_3_exc_layer_4_exc_weights_3_8s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im3_0_8s.csv');
layer_3_exc_layer_4_exc_weights_4_0s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im3_1_0s.csv');
layer_3_exc_layer_4_exc_weights_4_2s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im4_0_2s.csv');
layer_3_exc_layer_4_exc_weights_4_4s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im4_0_4s.csv');
layer_3_exc_layer_4_exc_weights_4_6s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im4_0_6000000000000001s.csv');
layer_3_exc_layer_4_exc_weights_4_8s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im4_0_8s.csv');
layer_3_exc_layer_4_exc_weights_5_0s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im4_1_0s.csv');
layer_3_exc_layer_4_exc_weights_5_2s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im5_0_2s.csv');
layer_3_exc_layer_4_exc_weights_5_4s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im5_0_4s.csv');
layer_3_exc_layer_4_exc_weights_5_6s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im5_0_6000000000000001s.csv');
layer_3_exc_layer_4_exc_weights_5_8s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im5_0_8s.csv');
layer_3_exc_layer_4_exc_weights_6_0s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im5_1_0s.csv');
layer_3_exc_layer_4_exc_weights_6_2s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im6_0_2s.csv');
layer_3_exc_layer_4_exc_weights_6_4s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im6_0_4s.csv');
layer_3_exc_layer_4_exc_weights_6_6s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im6_0_6000000000000001s.csv');
layer_3_exc_layer_4_exc_weights_6_8s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im6_0_8s.csv');
layer_3_exc_layer_4_exc_weights_7_0s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im6_1_0s.csv');
layer_3_exc_layer_4_exc_weights_7_2s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im7_0_2s.csv');
layer_3_exc_layer_4_exc_weights_7_4s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im7_0_4s.csv');
layer_3_exc_layer_4_exc_weights_7_6s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im7_0_6000000000000001s.csv');
layer_3_exc_layer_4_exc_weights_7_8s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im7_0_8s.csv');
layer_3_exc_layer_4_exc_weights_8_0s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_4_exc_weights_im7_1_0s.csv');
 
layer_1_exc_layer_1_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_1_exc_weights_0s.csv');
layer_1_exc_layer_1_inh_weights_0s = readmatrix('../output_data/simulation_5/layer_1_exc_layer_1_inh_weights_0s.csv');
layer_1_inh_layer_1_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_1_inh_layer_1_exc_weights_0s.csv');
layer_2_exc_layer_2_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_2_exc_weights_0s.csv');
layer_2_exc_layer_2_inh_weights_0s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_2_inh_weights_0s.csv');
layer_2_inh_layer_2_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_2_inh_layer_2_exc_weights_0s.csv');
layer_2_exc_layer_1_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_2_exc_layer_1_exc_weights_0s.csv');
layer_3_exc_layer_3_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_3_exc_weights_0s.csv');
layer_3_exc_layer_3_inh_weights_0s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_3_inh_weights_0s.csv');
layer_3_inh_layer_3_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_3_inh_layer_3_exc_weights_0s.csv');
layer_3_exc_layer_2_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_3_exc_layer_2_exc_weights_0s.csv');
layer_4_exc_layer_4_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_4_exc_layer_4_exc_weights_0s.csv');
layer_4_exc_layer_4_inh_weights_0s = readmatrix('../output_data/simulation_5/layer_4_exc_layer_4_inh_weights_0s.csv');
layer_4_inh_layer_4_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_4_inh_layer_4_exc_weights_0s.csv');
layer_4_exc_layer_3_exc_weights_0s = readmatrix('../output_data/simulation_5/layer_4_exc_layer_3_exc_weights_0s.csv');
%% separate spikes by stimulus

L1_exc_spikes_pre_test_idx = find(L1_exc_spikes(2,:)>8,1);
L1_exc_spikes_im1_idx = find(L1_exc_spikes(2,:)>10,1);
L1_exc_spikes_im1 = L1_exc_spikes(:,L1_exc_spikes_pre_test_idx:L1_exc_spikes_im1_idx);
L1_exc_spikes_im2_idx = find(L1_exc_spikes(2,:)>12,1);
L1_exc_spikes_im2 = L1_exc_spikes(:,L1_exc_spikes_im1_idx:L1_exc_spikes_im2_idx);
L1_exc_spikes_im3_idx = find(L1_exc_spikes(2,:)>14,1);
L1_exc_spikes_im3 = L1_exc_spikes(:,L1_exc_spikes_im2_idx:L1_exc_spikes_im3_idx);
L1_exc_spikes_im4_idx = find(L1_exc_spikes(2,:)>16,1);
L1_exc_spikes_im4 = L1_exc_spikes(:,L1_exc_spikes_im3_idx:L1_exc_spikes_im4_idx);
L1_exc_spikes_im5_idx = find(L1_exc_spikes(2,:)>18,1);
L1_exc_spikes_im5 = L1_exc_spikes(:,L1_exc_spikes_im4_idx:L1_exc_spikes_im5_idx);
L1_exc_spikes_im6_idx = find(L1_exc_spikes(2,:)>20,1);
L1_exc_spikes_im6 = L1_exc_spikes(:,L1_exc_spikes_im5_idx:L1_exc_spikes_im6_idx);
L1_exc_spikes_im7_idx = find(L1_exc_spikes(2,:)>22,1);
L1_exc_spikes_im7 = L1_exc_spikes(:,L1_exc_spikes_im6_idx:L1_exc_spikes_im7_idx);
L1_exc_spikes_im8_idx = find(L1_exc_spikes(2,:)>24,1);
L1_exc_spikes_im8 = L1_exc_spikes(:,L1_exc_spikes_im7_idx:L1_exc_spikes_im8_idx);
 
% L1_inh_spikes_pre_test_idx = find(L1_inh_spikes(2,:)>8,1);
% L1_inh_spikes_im1_idx = find(L1_inh_spikes(2,:)>10,1);
% L1_inh_spikes_im1 = L1_inh_spikes(:,L1_inh_spikes_pre_test_idx:L1_inh_spikes_im1_idx);
% L1_inh_spikes_im2_idx = find(L1_inh_spikes(2,:)>12,1);
% L1_inh_spikes_im2 = L1_inh_spikes(:,L1_inh_spikes_im1_idx:L1_inh_spikes_im2_idx);
% L1_inh_spikes_im3_idx = find(L1_inh_spikes(2,:)>14,1);
% L1_inh_spikes_im3 = L1_inh_spikes(:,L1_inh_spikes_im2_idx:L1_inh_spikes_im3_idx);
% L1_inh_spikes_im4_idx = find(L1_inh_spikes(2,:)>16,1);
% L1_inh_spikes_im4 = L1_inh_spikes(:,L1_inh_spikes_im3_idx:L1_inh_spikes_im4_idx);
% L1_inh_spikes_im5_idx = find(L1_inh_spikes(2,:)>18,1);
% L1_inh_spikes_im5 = L1_inh_spikes(:,L1_inh_spikes_im4_idx:L1_inh_spikes_im5_idx);
% L1_inh_spikes_im6_idx = find(L1_inh_spikes(2,:)>20,1);
% L1_inh_spikes_im6 = L1_inh_spikes(:,L1_inh_spikes_im5_idx:L1_inh_spikes_im6_idx);
% L1_inh_spikes_im7_idx = find(L1_inh_spikes(2,:)>22,1);
% L1_inh_spikes_im7 = L1_inh_spikes(:,L1_inh_spikes_im6_idx:L1_inh_spikes_im7_idx);
% L1_inh_spikes_im8_idx = find(L1_inh_spikes(2,:)>24,1);
% L1_inh_spikes_im8 = L1_inh_spikes(:,L1_inh_spikes_im7_idx:L1_inh_spikes_im8_idx);
 
L2_exc_spikes_pre_test_idx = find(L2_exc_spikes(2,:)>8,1);
L2_exc_spikes_im1_idx = find(L2_exc_spikes(2,:)>10,1);
L2_exc_spikes_im1 = L2_exc_spikes(:,L2_exc_spikes_pre_test_idx:L2_exc_spikes_im1_idx);
L2_exc_spikes_im2_idx = find(L2_exc_spikes(2,:)>12,1);
L2_exc_spikes_im2 = L2_exc_spikes(:,L2_exc_spikes_im1_idx:L2_exc_spikes_im2_idx);
L2_exc_spikes_im3_idx = find(L2_exc_spikes(2,:)>14,1);
L2_exc_spikes_im3 = L2_exc_spikes(:,L2_exc_spikes_im2_idx:L2_exc_spikes_im3_idx);
L2_exc_spikes_im4_idx = find(L2_exc_spikes(2,:)>16,1);
L2_exc_spikes_im4 = L2_exc_spikes(:,L2_exc_spikes_im3_idx:L2_exc_spikes_im4_idx);
L2_exc_spikes_im5_idx = find(L2_exc_spikes(2,:)>18,1);
L2_exc_spikes_im5 = L2_exc_spikes(:,L2_exc_spikes_im4_idx:L2_exc_spikes_im5_idx);
L2_exc_spikes_im6_idx = find(L2_exc_spikes(2,:)>20,1);
L2_exc_spikes_im6 = L2_exc_spikes(:,L2_exc_spikes_im5_idx:L2_exc_spikes_im6_idx);
L2_exc_spikes_im7_idx = find(L2_exc_spikes(2,:)>22,1);
L2_exc_spikes_im7 = L2_exc_spikes(:,L2_exc_spikes_im6_idx:L2_exc_spikes_im7_idx);
L2_exc_spikes_im8_idx = find(L2_exc_spikes(2,:)>24,1);
L2_exc_spikes_im8 = L2_exc_spikes(:,L2_exc_spikes_im7_idx:L2_exc_spikes_im8_idx);
 
L2_inh_spikes_pre_test_idx = find(L2_inh_spikes(2,:)>8,1);
L2_inh_spikes_im1_idx = find(L2_inh_spikes(2,:)>10,1);
L2_inh_spikes_im1 = L2_inh_spikes(:,L2_inh_spikes_pre_test_idx:L2_inh_spikes_im1_idx);
L2_inh_spikes_im2_idx = find(L2_inh_spikes(2,:)>12,1);
L2_inh_spikes_im2 = L2_inh_spikes(:,L2_inh_spikes_im1_idx:L2_inh_spikes_im2_idx);
L2_inh_spikes_im3_idx = find(L2_inh_spikes(2,:)>14,1);
L2_inh_spikes_im3 = L2_inh_spikes(:,L2_inh_spikes_im2_idx:L2_inh_spikes_im3_idx);
L2_inh_spikes_im4_idx = find(L2_inh_spikes(2,:)>16,1);
L2_inh_spikes_im4 = L2_inh_spikes(:,L2_inh_spikes_im3_idx:L2_inh_spikes_im4_idx);
L2_inh_spikes_im5_idx = find(L2_inh_spikes(2,:)>18,1);
L2_inh_spikes_im5 = L2_inh_spikes(:,L2_inh_spikes_im4_idx:L2_inh_spikes_im5_idx);
L2_inh_spikes_im6_idx = find(L2_inh_spikes(2,:)>20,1);
L2_inh_spikes_im6 = L2_inh_spikes(:,L2_inh_spikes_im5_idx:L2_inh_spikes_im6_idx);
L2_inh_spikes_im7_idx = find(L2_inh_spikes(2,:)>22,1);
L2_inh_spikes_im7 = L2_inh_spikes(:,L2_inh_spikes_im6_idx:L2_inh_spikes_im7_idx);
L2_inh_spikes_im8_idx = find(L2_inh_spikes(2,:)>24,1);
L2_inh_spikes_im8 = L2_inh_spikes(:,L2_inh_spikes_im7_idx:L2_inh_spikes_im8_idx);
 
L3_exc_spikes_pre_test_idx = find(L3_exc_spikes(2,:)>8,1);
L3_exc_spikes_im1_idx = find(L3_exc_spikes(2,:)>10,1);
L3_exc_spikes_im1 = L3_exc_spikes(:,L3_exc_spikes_pre_test_idx:L3_exc_spikes_im1_idx);
L3_exc_spikes_im2_idx = find(L3_exc_spikes(2,:)>12,1);
L3_exc_spikes_im2 = L3_exc_spikes(:,L3_exc_spikes_im1_idx:L3_exc_spikes_im2_idx);
L3_exc_spikes_im3_idx = find(L3_exc_spikes(2,:)>14,1);
L3_exc_spikes_im3 = L3_exc_spikes(:,L3_exc_spikes_im2_idx:L3_exc_spikes_im3_idx);
L3_exc_spikes_im4_idx = find(L3_exc_spikes(2,:)>16,1);
L3_exc_spikes_im4 = L3_exc_spikes(:,L3_exc_spikes_im3_idx:L3_exc_spikes_im4_idx);
L3_exc_spikes_im5_idx = find(L3_exc_spikes(2,:)>18,1);
L3_exc_spikes_im5 = L3_exc_spikes(:,L3_exc_spikes_im4_idx:L3_exc_spikes_im5_idx);
L3_exc_spikes_im6_idx = find(L3_exc_spikes(2,:)>20,1);
L3_exc_spikes_im6 = L3_exc_spikes(:,L3_exc_spikes_im5_idx:L3_exc_spikes_im6_idx);
L3_exc_spikes_im7_idx = find(L3_exc_spikes(2,:)>22,1);
L3_exc_spikes_im7 = L3_exc_spikes(:,L3_exc_spikes_im6_idx:L3_exc_spikes_im7_idx);
L3_exc_spikes_im8_idx = find(L3_exc_spikes(2,:)>24,1);
L3_exc_spikes_im8 = L3_exc_spikes(:,L3_exc_spikes_im7_idx:L3_exc_spikes_im8_idx);
 
L3_inh_spikes_pre_test_idx = find(L3_inh_spikes(2,:)>8,1);
L3_inh_spikes_im1_idx = find(L3_inh_spikes(2,:)>10,1);
L3_inh_spikes_im1 = L3_inh_spikes(:,L3_inh_spikes_pre_test_idx:L3_inh_spikes_im1_idx);
L3_inh_spikes_im2_idx = find(L3_inh_spikes(2,:)>12,1);
L3_inh_spikes_im2 = L3_inh_spikes(:,L3_inh_spikes_im1_idx:L3_inh_spikes_im2_idx);
L3_inh_spikes_im3_idx = find(L3_inh_spikes(2,:)>14,1);
L3_inh_spikes_im3 = L3_inh_spikes(:,L3_inh_spikes_im2_idx:L3_inh_spikes_im3_idx);
L3_inh_spikes_im4_idx = find(L3_inh_spikes(2,:)>16,1);
L3_inh_spikes_im4 = L3_inh_spikes(:,L3_inh_spikes_im3_idx:L3_inh_spikes_im4_idx);
L3_inh_spikes_im5_idx = find(L3_inh_spikes(2,:)>18,1);
L3_inh_spikes_im5 = L3_inh_spikes(:,L3_inh_spikes_im4_idx:L3_inh_spikes_im5_idx);
L3_inh_spikes_im6_idx = find(L3_inh_spikes(2,:)>20,1);
L3_inh_spikes_im6 = L3_inh_spikes(:,L3_inh_spikes_im5_idx:L3_inh_spikes_im6_idx);
L3_inh_spikes_im7_idx = find(L3_inh_spikes(2,:)>22,1);
L3_inh_spikes_im7 = L3_inh_spikes(:,L3_inh_spikes_im6_idx:L3_inh_spikes_im7_idx);
L3_inh_spikes_im8_idx = find(L3_inh_spikes(2,:)>24,1);
L3_inh_spikes_im8 = L3_inh_spikes(:,L3_inh_spikes_im7_idx:L3_inh_spikes_im8_idx);
 
L4_exc_spikes_pre_test_idx = find(L4_exc_spikes(2,:)>8,1);
L4_exc_spikes_im1_idx = find(L4_exc_spikes(2,:)>10,1);
L4_exc_spikes_im1 = L4_exc_spikes(:,L4_exc_spikes_pre_test_idx:L4_exc_spikes_im1_idx);
L4_exc_spikes_im2_idx = find(L4_exc_spikes(2,:)>12,1);
L4_exc_spikes_im2 = L4_exc_spikes(:,L4_exc_spikes_im1_idx:L4_exc_spikes_im2_idx);
L4_exc_spikes_im3_idx = find(L4_exc_spikes(2,:)>14,1);
L4_exc_spikes_im3 = L4_exc_spikes(:,L4_exc_spikes_im2_idx:L4_exc_spikes_im3_idx);
L4_exc_spikes_im4_idx = find(L4_exc_spikes(2,:)>16,1);
L4_exc_spikes_im4 = L4_exc_spikes(:,L4_exc_spikes_im3_idx:L4_exc_spikes_im4_idx);
L4_exc_spikes_im5_idx = find(L4_exc_spikes(2,:)>18,1);
L4_exc_spikes_im5 = L4_exc_spikes(:,L4_exc_spikes_im4_idx:L4_exc_spikes_im5_idx);
L4_exc_spikes_im6_idx = find(L4_exc_spikes(2,:)>20,1);
L4_exc_spikes_im6 = L4_exc_spikes(:,L4_exc_spikes_im5_idx:L4_exc_spikes_im6_idx);
L4_exc_spikes_im7_idx = find(L4_exc_spikes(2,:)>22,1);
L4_exc_spikes_im7 = L4_exc_spikes(:,L4_exc_spikes_im6_idx:L4_exc_spikes_im7_idx);
L4_exc_spikes_im8_idx = find(L4_exc_spikes(2,:)>24,1);
L4_exc_spikes_im8 = L4_exc_spikes(:,L4_exc_spikes_im7_idx:L4_exc_spikes_im8_idx);
 
L4_inh_spikes_pre_test_idx = find(L4_inh_spikes(2,:)>8,1);
L4_inh_spikes_im1_idx = find(L4_inh_spikes(2,:)>10,1);
L4_inh_spikes_im1 = L4_inh_spikes(:,L4_inh_spikes_pre_test_idx:L4_inh_spikes_im1_idx);
L4_inh_spikes_im2_idx = find(L4_inh_spikes(2,:)>12,1);
L4_inh_spikes_im2 = L4_inh_spikes(:,L4_inh_spikes_im1_idx:L4_inh_spikes_im2_idx);
L4_inh_spikes_im3_idx = find(L4_inh_spikes(2,:)>14,1);
L4_inh_spikes_im3 = L4_inh_spikes(:,L4_inh_spikes_im2_idx:L4_inh_spikes_im3_idx);
L4_inh_spikes_im4_idx = find(L4_inh_spikes(2,:)>16,1);
L4_inh_spikes_im4 = L4_inh_spikes(:,L4_inh_spikes_im3_idx:L4_inh_spikes_im4_idx);
L4_inh_spikes_im5_idx = find(L4_inh_spikes(2,:)>18,1);
L4_inh_spikes_im5 = L4_inh_spikes(:,L4_inh_spikes_im4_idx:L4_inh_spikes_im5_idx);
L4_inh_spikes_im6_idx = find(L4_inh_spikes(2,:)>20,1);
L4_inh_spikes_im6 = L4_inh_spikes(:,L4_inh_spikes_im5_idx:L4_inh_spikes_im6_idx);
L4_inh_spikes_im7_idx = find(L4_inh_spikes(2,:)>22,1);
L4_inh_spikes_im7 = L4_inh_spikes(:,L4_inh_spikes_im6_idx:L4_inh_spikes_im7_idx);
L4_inh_spikes_im8_idx = find(L4_inh_spikes(2,:)>24,1);
L4_inh_spikes_im8 = L4_inh_spikes(:,L4_inh_spikes_im7_idx:L4_inh_spikes_im8_idx);
%% calculate average firing rates for neurons for each image

[L1e_neurons_im1, L1e_count_im1, L1e_rates_im1] = rates(L1_exc_spikes_im1,4096,2);
[L1e_neurons_im2, L1e_count_im2, L1e_rates_im2] = rates(L1_exc_spikes_im2,4096,2);
[L1e_neurons_im3, L1e_count_im3, L1e_rates_im3] = rates(L1_exc_spikes_im3,4096,2);
[L1e_neurons_im4, L1e_count_im4, L1e_rates_im4] = rates(L1_exc_spikes_im4,4096,2);
[L1e_neurons_im5, L1e_count_im5, L1e_rates_im5] = rates(L1_exc_spikes_im5,4096,2);
[L1e_neurons_im6, L1e_count_im6, L1e_rates_im6] = rates(L1_exc_spikes_im6,4096,2);
[L1e_neurons_im7, L1e_count_im7, L1e_rates_im7] = rates(L1_exc_spikes_im7,4096,2);
[L1e_neurons_im8, L1e_count_im8, L1e_rates_im8] = rates(L1_exc_spikes_im8,4096,2);
 
% [L1i_neurons_im1, L1i_count_im1, L1i_rates_im1] = rates(L1_exc_spikes_im1,4096,2);
% [L1i_neurons_im2, L1i_count_im2, L1i_rates_im2] = rates(L1_exc_spikes_im2,4096,2);
% [L1i_neurons_im3, L1i_count_im3, L1i_rates_im3] = rates(L1_exc_spikes_im3,4096,2);
% [L1i_neurons_im4, L1i_count_im4, L1i_rates_im4] = rates(L1_exc_spikes_im4,4096,2);
% [L1i_neurons_im5, L1i_count_im5, L1i_rates_im5] = rates(L1_exc_spikes_im5,4096,2);
% [L1i_neurons_im6, L1i_count_im6, L1i_rates_im6] = rates(L1_exc_spikes_im6,4096,2);
% [L1i_neurons_im7, L1i_count_im7, L1i_rates_im7] = rates(L1_exc_spikes_im7,4096,2);
% [L1i_neurons_im8, L1i_count_im8, L1i_rates_im8] = rates(L1_exc_spikes_im8,4096,2);

[L2e_neurons_im1, L2e_count_im1, L2e_rates_im1] = rates(L2_exc_spikes_im1,4096,2);
[L2e_neurons_im2, L2e_count_im2, L2e_rates_im2] = rates(L2_exc_spikes_im2,4096,2);
[L2e_neurons_im3, L2e_count_im3, L2e_rates_im3] = rates(L2_exc_spikes_im3,4096,2);
[L2e_neurons_im4, L2e_count_im4, L2e_rates_im4] = rates(L2_exc_spikes_im4,4096,2);
[L2e_neurons_im5, L2e_count_im5, L2e_rates_im5] = rates(L2_exc_spikes_im5,4096,2);
[L2e_neurons_im6, L2e_count_im6, L2e_rates_im6] = rates(L2_exc_spikes_im6,4096,2);
[L2e_neurons_im7, L2e_count_im7, L2e_rates_im7] = rates(L2_exc_spikes_im7,4096,2);
[L2e_neurons_im8, L2e_count_im8, L2e_rates_im8] = rates(L2_exc_spikes_im8,4096,2);
 
[L2i_neurons_im1, L2i_count_im1, L2i_rates_im1] = rates(L2_exc_spikes_im1,4096,2);
[L2i_neurons_im2, L2i_count_im2, L2i_rates_im2] = rates(L2_exc_spikes_im2,4096,2);
[L2i_neurons_im3, L2i_count_im3, L2i_rates_im3] = rates(L2_exc_spikes_im3,4096,2);
[L2i_neurons_im4, L2i_count_im4, L2i_rates_im4] = rates(L2_exc_spikes_im4,4096,2);
[L2i_neurons_im5, L2i_count_im5, L2i_rates_im5] = rates(L2_exc_spikes_im5,4096,2);
[L2i_neurons_im6, L2i_count_im6, L2i_rates_im6] = rates(L2_exc_spikes_im6,4096,2);
[L2i_neurons_im7, L2i_count_im7, L2i_rates_im7] = rates(L2_exc_spikes_im7,4096,2);
[L2i_neurons_im8, L2i_count_im8, L2i_rates_im8] = rates(L2_exc_spikes_im8,4096,2);

[L3e_neurons_im1, L3e_count_im1, L3e_rates_im1] = rates(L3_exc_spikes_im1,4096,2);
[L3e_neurons_im2, L3e_count_im2, L3e_rates_im2] = rates(L3_exc_spikes_im2,4096,2);
[L3e_neurons_im3, L3e_count_im3, L3e_rates_im3] = rates(L3_exc_spikes_im3,4096,2);
[L3e_neurons_im4, L3e_count_im4, L3e_rates_im4] = rates(L3_exc_spikes_im4,4096,2);
[L3e_neurons_im5, L3e_count_im5, L3e_rates_im5] = rates(L3_exc_spikes_im5,4096,2);
[L3e_neurons_im6, L3e_count_im6, L3e_rates_im6] = rates(L3_exc_spikes_im6,4096,2);
[L3e_neurons_im7, L3e_count_im7, L3e_rates_im7] = rates(L3_exc_spikes_im7,4096,2);
[L3e_neurons_im8, L3e_count_im8, L3e_rates_im8] = rates(L3_exc_spikes_im8,4096,2);

[L3i_neurons_im1, L3i_count_im1, L3i_rates_im1] = rates(L3_exc_spikes_im1,4096,2);
[L3i_neurons_im2, L3i_count_im2, L3i_rates_im2] = rates(L3_exc_spikes_im2,4096,2);
[L3i_neurons_im3, L3i_count_im3, L3i_rates_im3] = rates(L3_exc_spikes_im3,4096,2);
[L3i_neurons_im4, L3i_count_im4, L3i_rates_im4] = rates(L3_exc_spikes_im4,4096,2);
[L3i_neurons_im5, L3i_count_im5, L3i_rates_im5] = rates(L3_exc_spikes_im5,4096,2);
[L3i_neurons_im6, L3i_count_im6, L3i_rates_im6] = rates(L3_exc_spikes_im6,4096,2);
[L3i_neurons_im7, L3i_count_im7, L3i_rates_im7] = rates(L3_exc_spikes_im7,4096,2);
[L3i_neurons_im8, L3i_count_im8, L3i_rates_im8] = rates(L3_exc_spikes_im8,4096,2);

[L4e_neurons_im1, L4e_count_im1, L4e_rates_im1] = rates(L4_exc_spikes_im1,4096,2);
[L4e_neurons_im2, L4e_count_im2, L4e_rates_im2] = rates(L4_exc_spikes_im2,4096,2);
[L4e_neurons_im3, L4e_count_im3, L4e_rates_im3] = rates(L4_exc_spikes_im3,4096,2);
[L4e_neurons_im4, L4e_count_im4, L4e_rates_im4] = rates(L4_exc_spikes_im4,4096,2);
[L4e_neurons_im5, L4e_count_im5, L4e_rates_im5] = rates(L4_exc_spikes_im5,4096,2);
[L4e_neurons_im6, L4e_count_im6, L4e_rates_im6] = rates(L4_exc_spikes_im6,4096,2);
[L4e_neurons_im7, L4e_count_im7, L4e_rates_im7] = rates(L4_exc_spikes_im7,4096,2);
[L4e_neurons_im8, L4e_count_im8, L4e_rates_im8] = rates(L4_exc_spikes_im8,4096,2);

[L4i_neurons_im1, L4i_count_im1, L4i_rates_im1] = rates(L4_exc_spikes_im1,4096,2);
[L4i_neurons_im2, L4i_count_im2, L4i_rates_im2] = rates(L4_exc_spikes_im2,4096,2);
[L4i_neurons_im3, L4i_count_im3, L4i_rates_im3] = rates(L4_exc_spikes_im3,4096,2);
[L4i_neurons_im4, L4i_count_im4, L4i_rates_im4] = rates(L4_exc_spikes_im4,4096,2);
[L4i_neurons_im5, L4i_count_im5, L4i_rates_im5] = rates(L4_exc_spikes_im5,4096,2);
[L4i_neurons_im6, L4i_count_im6, L4i_rates_im6] = rates(L4_exc_spikes_im6,4096,2);
[L4i_neurons_im7, L4i_count_im7, L4i_rates_im7] = rates(L4_exc_spikes_im7,4096,2);
[L4i_neurons_im8, L4i_count_im8, L4i_rates_im8] = rates(L4_exc_spikes_im8,4096,2);
%% raster plots

figure()
scatter(L3_exc_spikes_test(2,:),L3_exc_spikes_test(1,:))
%% plot average firing rates
[L1_exc_neurons, L1_exc_count, L1_exc_rates] = rates(L1_exc_spikes,4096,16);
% [L1_inh_neurons, L1_inh_count, L1_inh_rates] = rates(L1_inh_spikes,4096,16);
[L2_exc_neurons, L2_exc_count, L2_exc_rates] = rates(L2_exc_spikes,4096,16);
[L2_inh_neurons, L2_inh_count, L2_inh_rates] = rates(L2_inh_spikes,4096,16);
[L3_exc_neurons, L3_exc_count, L3_exc_rates] = rates(L3_exc_spikes,4096,16);
[L3_inh_neurons, L3_inh_count, L3_inh_rates] = rates(L3_inh_spikes,4096,16);
[L4_exc_neurons, L4_exc_count, L4_exc_rates] = rates(L4_exc_spikes,4096,16);
[L4_inh_neurons, L4_inh_count, L4_inh_rates] = rates(L4_inh_spikes,4096,16);

subplot(4,2,1)
histogram(L1_exc_rates,50,'EdgeColor','none','FaceColor',[rand, rand, rand],'FaceAlpha',0.8);
xlabel('ave. firing rate (Hz)')
ylabel('cell count')
title('L1 exc.')
grid on

L1_inh_rates = zeros(4096,1);
subplot(4,2,2)
histogram(L1_inh_rates,50,'EdgeColor','none','FaceColor',[rand, rand, rand],'FaceAlpha',0.8);
xlabel('ave. firing rate (Hz)')
ylabel('cell count')
xlim([0,1])
title('L3 exc.')
grid on

subplot(4,2,3)
histogram(L2_exc_rates,50,'EdgeColor','none','FaceColor',[rand, rand, rand],'FaceAlpha',0.8);
xlabel('ave. firing rate (Hz)')
ylabel('cell count')
title('L2 exc.')
grid on

subplot(4,2,4)
histogram(L2_inh_rates,50,'EdgeColor','none','FaceColor',[rand, rand, rand],'FaceAlpha',0.8);
xlabel('ave. firing rate (Hz)')
ylabel('cell count')
title('L2 inh.')
grid on

subplot(4,2,5)
histogram(L3_exc_rates,50,'EdgeColor','none','FaceColor',[rand, rand, rand],'FaceAlpha',0.8);
xlabel('ave. firing rate (Hz)')
ylabel('cell count')
title('L3 exc.')
grid on

subplot(4,2,6)
histogram(L3_inh_rates,50,'EdgeColor','none','FaceColor',[rand, rand, rand],'FaceAlpha',0.8);
xlabel('ave. firing rate (Hz)')
ylabel('cell count')
title('L3 inh.')
grid on

subplot(4,2,7)
histogram(L4_exc_rates,50,'EdgeColor','none','FaceColor',[rand, rand, rand],'FaceAlpha',0.8);
xlabel('ave. firing rate (Hz)')
ylabel('cell count')
title('L4 exc.')
grid on

subplot(4,2,8)
histogram(L4_inh_rates,50,'EdgeColor','none','FaceColor',[rand, rand, rand],'FaceAlpha',0.8);
xlabel('ave. firing rate (Hz)')
ylabel('cell count')
title('L4 inh.')
grid on
%% selective neurons

% for num = [1000:1100]
%     figure()
%     hold on
%     bar([1:8],[L3e_rates_im1(num),L3e_rates_im2(num),L3e_rates_im3(num),L3e_rates_im4(num),L3e_rates_im5(num),L3e_rates_im6(num),L3e_rates_im7(num),L3e_rates_im8(num)],0.9,'EdgeColor','none','FaceColor',[rand, rand, rand],'FaceAlpha',0.8)
%     title(['exc. LIF neuron #',num2str(num),' in layer 3'])
%     ylabel('average firing rate (Hz)')
%     xlabel('image')
%     xticks([1:8])
%     xlim([0.4,8.6])
%     grid on
% end

% get average firing rates for specific image sets (see lab notebook)
L3e_rates_set1 = (L3e_rates_im8+L3e_rates_im6+L3e_rates_im1+L3e_rates_im3)/4;
L3e_rates_set2 = (L3e_rates_im8+L3e_rates_im6+L3e_rates_im1+L3e_rates_im2)/4;
L3e_rates_set3 = (L3e_rates_im8+L3e_rates_im6+L3e_rates_im2+L3e_rates_im7)/4;
L3e_rates_set4 = (L3e_rates_im8+L3e_rates_im2+L3e_rates_im7+L3e_rates_im4)/4;

selective_concave_top = [];

neuron_num = [1:4096];

for num = neuron_num
    if L3e_rates_set1(num) > L3e_rates_set2(num) && L3e_rates_set2(num) > L3e_rates_set3(num) && L3e_rates_set3(num) > L3e_rates_set4(num)
        selective_concave_top(end+1) = num;
    end
end

for num = selective_concave_top
    figure()
    hold on
    bar([1:4],[L3e_rates_set1(num),L3e_rates_set2(num),L3e_rates_set3(num),L3e_rates_set4(num)],0.9,'EdgeColor','none','FaceColor',[rand, rand, rand],'FaceAlpha',0.8)
    title(['exc. LIF neuron #',num2str(num),' in layer 3'])
    ylabel('average firing rate (Hz)')
    xlabel('image set')
    set(gca,'xtick',1:4);
    set(gca,'xticklabel',{'A', 'B', 'C','D'},'fontsize',8)
    xlim([0.4,4.6])
    grid on
end
%% trace selective neurons to inputs

l3_neuron = 667;
l2_neuron = 1055;
l1_neuron = 9105;
l2_inputs = trace_to_inputs(l3_neuron,layer_2_exc_layer_3_exc_weights_8_0s);
l1_inputs = trace_to_inputs(l2_neuron,layer_1_exc_layer_2_exc_weights_8_0s);
l0_inputs = trace_to_inputs(l1_neuron,layer_0_layer_1_exc_weights_8_0s);

%% evolution of weights of relevant synapses

synapse_num = 12345;
weight = [layer_3_exc_layer_4_exc_weights_0s(5,synapse_num),layer_2_exc_layer_3_exc_weights_0_2s(5,synapse_num),layer_2_exc_layer_3_exc_weights_0_4s(5,synapse_num),layer_2_exc_layer_3_exc_weights_0_6s(5,synapse_num),layer_2_exc_layer_3_exc_weights_0_8s(5,synapse_num),layer_2_exc_layer_3_exc_weights_1_0s(5,synapse_num),layer_2_exc_layer_3_exc_weights_1_2s(5,synapse_num),layer_2_exc_layer_3_exc_weights_1_4s(5,synapse_num),layer_2_exc_layer_3_exc_weights_1_6s(5,synapse_num),layer_2_exc_layer_3_exc_weights_1_8s(5,synapse_num),layer_2_exc_layer_3_exc_weights_2_0s(5,synapse_num),layer_2_exc_layer_3_exc_weights_2_2s(5,synapse_num),layer_2_exc_layer_3_exc_weights_2_4s(5,synapse_num),layer_2_exc_layer_3_exc_weights_2_6s(5,synapse_num),layer_2_exc_layer_3_exc_weights_2_8s(5,synapse_num),layer_2_exc_layer_3_exc_weights_3_0s(5,synapse_num),layer_2_exc_layer_3_exc_weights_3_2s(5,synapse_num),layer_2_exc_layer_3_exc_weights_3_4s(5,synapse_num),layer_2_exc_layer_3_exc_weights_3_6s(5,synapse_num),layer_2_exc_layer_3_exc_weights_3_8s(5,synapse_num),layer_2_exc_layer_3_exc_weights_4_0s(5,synapse_num),layer_2_exc_layer_3_exc_weights_4_2s(5,synapse_num),layer_2_exc_layer_3_exc_weights_4_4s(5,synapse_num),layer_2_exc_layer_3_exc_weights_4_6s(5,synapse_num),layer_2_exc_layer_3_exc_weights_4_8s(5,synapse_num),layer_2_exc_layer_3_exc_weights_5_0s(5,synapse_num),layer_2_exc_layer_3_exc_weights_5_2s(5,synapse_num),layer_2_exc_layer_3_exc_weights_5_4s(5,synapse_num),layer_2_exc_layer_3_exc_weights_5_6s(5,synapse_num),layer_2_exc_layer_3_exc_weights_5_8s(5,synapse_num),layer_2_exc_layer_3_exc_weights_6_0s(5,synapse_num),layer_2_exc_layer_3_exc_weights_6_2s(5,synapse_num),layer_2_exc_layer_3_exc_weights_6_4s(5,synapse_num),layer_2_exc_layer_3_exc_weights_6_6s(5,synapse_num),layer_2_exc_layer_3_exc_weights_6_8s(5,synapse_num),layer_2_exc_layer_3_exc_weights_7_0s(5,synapse_num),layer_2_exc_layer_3_exc_weights_7_2s(5,synapse_num),layer_2_exc_layer_3_exc_weights_7_4s(5,synapse_num),layer_2_exc_layer_3_exc_weights_7_6s(5,synapse_num),layer_2_exc_layer_3_exc_weights_7_8s(5,synapse_num),layer_2_exc_layer_3_exc_weights_8_0s(5,synapse_num)];
time = [0:40]/5;
figure()
plot(time,weight)
grid on
xlabel('time (s)')
ylabel('weight (S)')
title('evolution of weight for synapse ',synapse_num)
%% function definitions

function [neurons, N_spikes, rates] = rates(spikes,N_neurons,simulation_time)
    idx = spikes(1,:)+1;
    neurons = [1:N_neurons];
    N_spikes = zeros(1,length(neurons));
    for i=idx
        N_spikes(i) = N_spikes(i)+1;
    end
    for i=neurons
        rates(i) = N_spikes(i)/simulation_time;
    end
end

% function to trace synapses from a neuron to neurons it's connected to in
% the previous layer
function inputs = trace_to_inputs(neuron_num,synapse_data)
    % get location of neuron
    x_loc = mod(neuron_num,64)*5e-5;
    y_loc = floor(neuron_num/64)*5e-5;
    % find synapses from previous layer neurons to this neuron
    syn_idx = find(round(synapse_data(3,:),4) == round(x_loc,4) & round(synapse_data(4,:),4) == round(y_loc,4));
    % store index, weight and x and y locations of all connected neurons
    inputs = [syn_idx;synapse_data(5,syn_idx);synapse_data(1,syn_idx);synapse_data(2,syn_idx)];
    % sort by weights in descending order 
    inputs = transpose(inputs);
    inputs = sortrows(inputs,2,'descend');
    inputs = transpose(inputs);
end

% function to trace synapses from a neuron to neurons it's connected to in
% the previous layer
function inputs = trace_to_inputs_poisson(neuron_num,synapse_data)
    % get location of neuron
    x_loc = mod(neuron_num,64)*5e-5;
    y_loc = floor(neuron_num/64)*5e-5;
    % find synapses from previous layer neurons to this neuron
    syn_idx = find(round(synapse_data(3,:),4) == round(x_loc,4) & round(synapse_data(4,:),4) == round(y_loc,4));
    % store index, weight and x and y locations and filter number of all connected neurons
    inputs = [syn_idx;synapse_data(5,syn_idx);synapse_data(1,syn_idx);synapse_data(2,syn_idx);synapse_data(6,syn_idx)];
end














