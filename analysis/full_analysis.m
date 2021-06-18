clear all; close all; clc
simulation_num = '20';
%% read spike data
% spike data
L0_spikes = readmatrix(sprintf('../output_data/simulation_%s/layer_0_full_spikes.csv',simulation_num));
L0_spikes_train_idx = find(L0_spikes(2,:)>16,1);
L0_spikes_train = L0_spikes(:,1:L0_spikes_train_idx);
L0_spikes_test_idx = find(L0_spikes(2,:)>32,1);
L0_spikes_test = L0_spikes(:,L0_spikes_train_idx:end);%L0_spikes_test_idx);
L1_exc_spikes = readmatrix(sprintf('../output_data/simulation_%s/layer_1_excitatory_full_spikes.csv',simulation_num));
L1_exc_spikes_train_idx = find(L1_exc_spikes(2,:)>16,1);
L1_exc_spikes_train = L1_exc_spikes(:,1:L1_exc_spikes_train_idx);
L1_exc_spikes_test_idx = find(L1_exc_spikes(2,:)>32,1);
L1_exc_spikes_test = L1_exc_spikes(:,L1_exc_spikes_train_idx:end);%L1_exc_spikes_test_idx);
L1_inh_spikes = readmatrix(sprintf('../output_data/simulation_%s/layer_1_inhibitory_full_spikes.csv',simulation_num));
L1_inh_spikes_train_idx = find(L1_inh_spikes(2,:)>16,1);
L1_inh_spikes_train = L1_inh_spikes(:,1:L1_inh_spikes_train_idx);
L1_inh_spikes_test_idx = find(L1_inh_spikes(2,:)>32,1);
L1_inh_spikes_test = L1_inh_spikes(:,L1_inh_spikes_train_idx:end);%L1_inh_spikes_test_idx);
L2_exc_spikes = readmatrix(sprintf('../output_data/simulation_%s/layer_2_excitatory_full_spikes.csv',simulation_num));
L2_exc_spikes_train_idx = find(L2_exc_spikes(2,:)>16,1);
L2_exc_spikes_train = L2_exc_spikes(:,1:L2_exc_spikes_train_idx);
L2_exc_spikes_test_idx = find(L2_exc_spikes(2,:)>32,1);
L2_exc_spikes_test = L2_exc_spikes(:,L2_exc_spikes_train_idx:end);%L2_exc_spikes_test_idx);
L2_inh_spikes = readmatrix(sprintf('../output_data/simulation_%s/layer_2_inhibitory_full_spikes.csv',simulation_num));
L2_inh_spikes_train_idx = find(L2_inh_spikes(2,:)>16,1);
L2_inh_spikes_train = L2_inh_spikes(:,1:L2_inh_spikes_train_idx);
L2_inh_spikes_test_idx = find(L2_inh_spikes(2,:)>32,1);
L2_inh_spikes_test = L2_inh_spikes(:,L2_inh_spikes_train_idx:end);%L2_inh_spikes_test_idx);
L3_exc_spikes = readmatrix(sprintf('../output_data/simulation_%s/layer_3_excitatory_full_spikes.csv',simulation_num));
L3_exc_spikes_train_idx = find(L3_exc_spikes(2,:)>16,1);
L3_exc_spikes_train = L3_exc_spikes(:,1:L3_exc_spikes_train_idx);
L3_exc_spikes_test_idx = find(L3_exc_spikes(2,:)>32,1);
L3_exc_spikes_test = L3_exc_spikes(:,L3_exc_spikes_train_idx:end);%L3_exc_spikes_test_idx);
L3_inh_spikes = readmatrix(sprintf('../output_data/simulation_%s/layer_3_inhibitory_full_spikes.csv',simulation_num));
L3_inh_spikes_train_idx = find(L3_inh_spikes(2,:)>16,1);
L3_inh_spikes_train = L3_inh_spikes(:,1:L3_inh_spikes_train_idx);
L3_inh_spikes_test_idx = find(L3_inh_spikes(2,:)>32,1);
L3_inh_spikes_test = L3_inh_spikes(:,L3_inh_spikes_train_idx:end);%L3_inh_spikes_test_idx);
L4_exc_spikes = readmatrix(sprintf('../output_data/simulation_%s/layer_4_excitatory_full_spikes.csv',simulation_num));
L4_exc_spikes_train_idx = find(L4_exc_spikes(2,:)>16,1);
L4_exc_spikes_train = L4_exc_spikes(:,1:L4_exc_spikes_train_idx);
L4_exc_spikes_test_idx = find(L4_exc_spikes(2,:)>32,1);
L4_exc_spikes_test = L4_exc_spikes(:,L4_exc_spikes_train_idx:end);%L4_exc_spikes_test_idx);
L4_inh_spikes = readmatrix(sprintf('../output_data/simulation_%s/layer_4_inhibitory_full_spikes.csv',simulation_num));
L4_inh_spikes_train_idx = find(L4_inh_spikes(2,:)>16,1);
L4_inh_spikes_train = L4_inh_spikes(:,1:L4_inh_spikes_train_idx);
L4_inh_spikes_test_idx = find(L4_inh_spikes(2,:)>32,1);
L4_inh_spikes_test = L4_inh_spikes(:,L4_inh_spikes_train_idx:end);%L4_inh_spikes_test_idx);
%% raster plots
figure()
scatter(L3_exc_spikes_test(2,:)*1000,L3_exc_spikes_test(1,:),'*','MarkerEdgeColor','r')
xlabel('time (ms)')
ylabel('neuron index')
title('Raster plot for layer 3 excitatory neurons')
grid on
%% read weight data
% layer_0_layer_1_exc_weights_0s = readmatrix(sprintf(‘../output_data/simulation_13/layer_0_layer_1_exc_weights_0s.csv’,simulation_num));
layer_0_layer_1_exc_weights_8s_idx_pre = readmatrix(sprintf('../output_data/simulation_%s/layer_0_layer_1_exc_weights_8s_idx_pre.csv',simulation_num));
layer_0_layer_1_exc_weights_8s_idx_post = readmatrix(sprintf('../output_data/simulation_%s/layer_0_layer_1_exc_weights_8s_idx_post.csv',simulation_num));
layer_0_layer_1_exc_weights_8s_x_pre = readmatrix(sprintf('../output_data/simulation_%s/layer_0_layer_1_exc_weights_8s_x_pre.csv',simulation_num));
layer_0_layer_1_exc_weights_8s_y_pre = readmatrix(sprintf('../output_data/simulation_%s/layer_0_layer_1_exc_weights_8s_y_pre.csv',simulation_num));
layer_0_layer_1_exc_weights_8s_f_pre = readmatrix(sprintf('../output_data/simulation_%s/layer_0_layer_1_exc_weights_8s_f_pre.csv',simulation_num));
layer_0_layer_1_exc_weights_8s_w = readmatrix(sprintf('../output_data/simulation_%s/layer_0_layer_1_exc_weights_8s_w.csv',simulation_num));
% layer_0_layer_1_exc_weights_0s_w = readmatrix(sprintf(‘../output_data/simulation_%s/layer_0_layer_1_exc_weights_0s_w.csv',simulation_num));
 
layer_0_layer_1_exc_weights_8s = [layer_0_layer_1_exc_weights_8s_idx_pre;layer_0_layer_1_exc_weights_8s_idx_post;layer_0_layer_1_exc_weights_8s_x_pre;layer_0_layer_1_exc_weights_8s_y_pre;layer_0_layer_1_exc_weights_8s_f_pre;layer_0_layer_1_exc_weights_8s_w];
% layer_1_exc_layer_2_exc_weights_0s = readmatrix(sprintf(‘../output_data/simulation_%s/layer_1_exc_layer_2_exc_weights_0s.csv',simulation_num));
layer_1_exc_layer_2_exc_weights_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_1_exc_layer_2_exc_weights_8s.csv',simulation_num)); 
% layer_2_exc_layer_3_exc_weights_0s = readmatrix(sprintf(‘../output_data/simulation_%s/layer_2_exc_layer_3_exc_weights_0s.csv',simulation_num));
layer_2_exc_layer_3_exc_weights_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_2_exc_layer_3_exc_weights_8s.csv',simulation_num));
% layer_3_exc_layer_4_exc_weights_0s = readmatrix(sprintf(‘../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_0s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_8s.csv',simulation_num));
 
%  
% layer_1_exc_layer_1_exc_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_1_exc_layer_1_exc_weights_0s.csv',simulation_num));
% layer_1_exc_layer_1_inh_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_1_exc_layer_1_inh_weights_0s.csv',simulation_num));
% layer_1_inh_layer_1_exc_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_1_inh_layer_1_exc_weights_0s.csv,simulation_num));
% layer_2_exc_layer_2_exc_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_2_exc_layer_2_exc_weights_0s.csv’,simulation_num));
% layer_2_exc_layer_2_inh_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_2_exc_layer_2_inh_weights_0s.csv'’,simulation_num));
% layer_2_inh_layer_2_exc_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_2_inh_layer_2_exc_weights_0s.csv',simulation_num));
% layer_2_exc_layer_1_exc_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_2_exc_layer_1_exc_weights_0s.csv',simulation_num));
% layer_3_exc_layer_3_exc_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_3_exc_weights_0s.csv',simulation_num));
% layer_3_exc_layer_3_inh_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_3_inh_weights_0s.csv',simulation_num));
% layer_3_inh_layer_3_exc_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_inh_layer_3_exc_weights_0s.csv',simulation_num));
% layer_3_exc_layer_2_exc_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_2_exc_weights_0s.csv',simulation_num));
layer_4_exc_layer_4_exc_weights_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_4_exc_layer_4_exc_weights_8s.csv',simulation_num));
% layer_4_exc_layer_4_inh_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_4_exc_layer_4_inh_weights_0s.csv,simulation_num));
% layer_4_inh_layer_4_exc_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_4_inh_layer_4_exc_weights_0s.csv’,simulation_num));
% layer_4_exc_layer_3_exc_weights_0s = readmatrix(sprintf('../output_data/simulation_%s/layer_4_exc_layer_3_exc_weights_0s.csv',simulation_num));
%% separate spikes by stimulus
L1_exc_spikes_start_idx = find(L1_exc_spikes(2,:)==0,1);
L1_exc_spikes_train_im1_idx = find(L1_exc_spikes(2,:)>1,1);
L1_exc_spikes_train_im1 = L1_exc_spikes(:,L1_exc_spikes_start_idx:L1_exc_spikes_train_im1_idx);
L1_exc_spikes_train_im2_idx = find(L1_exc_spikes(2,:)>2,1);
L1_exc_spikes_train_im2 = L1_exc_spikes(:,L1_exc_spikes_train_im1_idx:L1_exc_spikes_train_im2_idx);
L1_exc_spikes_train_im3_idx = find(L1_exc_spikes(2,:)>3,1);
L1_exc_spikes_train_im3 = L1_exc_spikes(:,L1_exc_spikes_train_im2_idx:L1_exc_spikes_train_im3_idx);
L1_exc_spikes_train_im4_idx = find(L1_exc_spikes(2,:)>4,1);
L1_exc_spikes_train_im4 = L1_exc_spikes(:,L1_exc_spikes_train_im3_idx:L1_exc_spikes_train_im4_idx);
L1_exc_spikes_train_im5_idx = find(L1_exc_spikes(2,:)>5,1);
L1_exc_spikes_train_im5 = L1_exc_spikes(:,L1_exc_spikes_train_im4_idx:L1_exc_spikes_train_im5_idx);
L1_exc_spikes_train_im6_idx = find(L1_exc_spikes(2,:)>6,1);
L1_exc_spikes_train_im6 = L1_exc_spikes(:,L1_exc_spikes_train_im5_idx:L1_exc_spikes_train_im6_idx);
L1_exc_spikes_train_im7_idx = find(L1_exc_spikes(2,:)>7,1);
L1_exc_spikes_train_im7 = L1_exc_spikes(:,L1_exc_spikes_train_im6_idx:L1_exc_spikes_train_im7_idx);
L1_exc_spikes_train_im8_idx = find(L1_exc_spikes(2,:)>8,1);
L1_exc_spikes_train_im8 = L1_exc_spikes(:,L1_exc_spikes_train_im7_idx:L1_exc_spikes_train_im8_idx);
L1_exc_spikes_train_im9_idx = find(L1_exc_spikes(2,:)>9,1);
L1_exc_spikes_train_im9 = L1_exc_spikes(:,L1_exc_spikes_train_im8_idx:L1_exc_spikes_train_im9_idx);
L1_exc_spikes_train_im10_idx = find(L1_exc_spikes(2,:)>10,1);
L1_exc_spikes_train_im10 = L1_exc_spikes(:,L1_exc_spikes_train_im9_idx:L1_exc_spikes_train_im10_idx);
L1_exc_spikes_train_im11_idx = find(L1_exc_spikes(2,:)>11,1);
L1_exc_spikes_train_im11 = L1_exc_spikes(:,L1_exc_spikes_train_im10_idx:L1_exc_spikes_train_im11_idx);
L1_exc_spikes_train_im12_idx = find(L1_exc_spikes(2,:)>12,1);
L1_exc_spikes_train_im12 = L1_exc_spikes(:,L1_exc_spikes_train_im11_idx:L1_exc_spikes_train_im12_idx);
L1_exc_spikes_train_im13_idx = find(L1_exc_spikes(2,:)>13,1);
L1_exc_spikes_train_im13 = L1_exc_spikes(:,L1_exc_spikes_train_im12_idx:L1_exc_spikes_train_im13_idx);
L1_exc_spikes_train_im14_idx = find(L1_exc_spikes(2,:)>14,1);
L1_exc_spikes_train_im14 = L1_exc_spikes(:,L1_exc_spikes_train_im13_idx:L1_exc_spikes_train_im14_idx);
L1_exc_spikes_train_im15_idx = find(L1_exc_spikes(2,:)>15,1);
L1_exc_spikes_train_im15 = L1_exc_spikes(:,L1_exc_spikes_train_im14_idx:L1_exc_spikes_train_im15_idx);
L1_exc_spikes_train_im16_idx = find(L1_exc_spikes(2,:)>16,1);
L1_exc_spikes_train_im16 = L1_exc_spikes(:,L1_exc_spikes_train_im15_idx:L1_exc_spikes_train_im16_idx);
 
L1_exc_spikes_test_im1_idx = find(L1_exc_spikes(2,:)>17,1);
L1_exc_spikes_test_im1 = L1_exc_spikes(:,L1_exc_spikes_train_im16_idx:L1_exc_spikes_test_im1_idx);
L1_exc_spikes_test_im2_idx = find(L1_exc_spikes(2,:)>18,1);
L1_exc_spikes_test_im2 = L1_exc_spikes(:,L1_exc_spikes_test_im1_idx:L1_exc_spikes_test_im2_idx);
L1_exc_spikes_test_im3_idx = find(L1_exc_spikes(2,:)>19,1);
L1_exc_spikes_test_im3 = L1_exc_spikes(:,L1_exc_spikes_test_im2_idx:L1_exc_spikes_test_im3_idx);
L1_exc_spikes_test_im4_idx = find(L1_exc_spikes(2,:)>20,1);
L1_exc_spikes_test_im4 = L1_exc_spikes(:,L1_exc_spikes_test_im3_idx:L1_exc_spikes_test_im4_idx);
L1_exc_spikes_test_im5_idx = find(L1_exc_spikes(2,:)>21,1);
L1_exc_spikes_test_im5 = L1_exc_spikes(:,L1_exc_spikes_test_im4_idx:L1_exc_spikes_test_im5_idx);
L1_exc_spikes_test_im6_idx = find(L1_exc_spikes(2,:)>22,1);
L1_exc_spikes_test_im6 = L1_exc_spikes(:,L1_exc_spikes_test_im5_idx:L1_exc_spikes_test_im6_idx);
L1_exc_spikes_test_im7_idx = find(L1_exc_spikes(2,:)>23,1);
L1_exc_spikes_test_im7 = L1_exc_spikes(:,L1_exc_spikes_test_im6_idx:L1_exc_spikes_test_im7_idx);
L1_exc_spikes_test_im8_idx = find(L1_exc_spikes(2,:)>24,1);
L1_exc_spikes_test_im8 = L1_exc_spikes(:,L1_exc_spikes_test_im7_idx:L1_exc_spikes_test_im8_idx);
L1_exc_spikes_test_im9_idx = find(L1_exc_spikes(2,:)>25,1);
L1_exc_spikes_test_im9 = L1_exc_spikes(:,L1_exc_spikes_test_im8_idx:L1_exc_spikes_test_im9_idx);
L1_exc_spikes_test_im10_idx = find(L1_exc_spikes(2,:)>26,1);
L1_exc_spikes_test_im10 = L1_exc_spikes(:,L1_exc_spikes_test_im9_idx:L1_exc_spikes_test_im10_idx);
L1_exc_spikes_test_im11_idx = find(L1_exc_spikes(2,:)>27,1);
L1_exc_spikes_test_im11 = L1_exc_spikes(:,L1_exc_spikes_test_im10_idx:L1_exc_spikes_test_im11_idx);
L1_exc_spikes_test_im12_idx = find(L1_exc_spikes(2,:)>28,1);
L1_exc_spikes_test_im12 = L1_exc_spikes(:,L1_exc_spikes_test_im11_idx:L1_exc_spikes_test_im12_idx);
L1_exc_spikes_test_im13_idx = find(L1_exc_spikes(2,:)>29,1);
L1_exc_spikes_test_im13 = L1_exc_spikes(:,L1_exc_spikes_test_im12_idx:L1_exc_spikes_test_im13_idx);
L1_exc_spikes_test_im14_idx = find(L1_exc_spikes(2,:)>30,1);
L1_exc_spikes_test_im14 = L1_exc_spikes(:,L1_exc_spikes_test_im13_idx:L1_exc_spikes_test_im14_idx);
L1_exc_spikes_test_im15_idx = find(L1_exc_spikes(2,:)>31,1);
L1_exc_spikes_test_im15 = L1_exc_spikes(:,L1_exc_spikes_test_im14_idx:L1_exc_spikes_test_im15_idx);
L1_exc_spikes_test_im16_idx = find(L1_exc_spikes(2,:)>32,1);
L1_exc_spikes_test_im16 = L1_exc_spikes(:,L1_exc_spikes_test_im15_idx:L1_exc_spikes_test_im16_idx);
 
L1_inh_spikes_start_idx = find(L1_inh_spikes(2,:)==0,1);
L1_inh_spikes_train_im1_idx = find(L1_inh_spikes(2,:)>1,1);
L1_inh_spikes_train_im1 = L1_inh_spikes(:,L1_inh_spikes_start_idx:L1_inh_spikes_train_im1_idx);
L1_inh_spikes_train_im2_idx = find(L1_inh_spikes(2,:)>2,1);
L1_inh_spikes_train_im2 = L1_inh_spikes(:,L1_inh_spikes_train_im1_idx:L1_inh_spikes_train_im2_idx);
L1_inh_spikes_train_im3_idx = find(L1_inh_spikes(2,:)>3,1);
L1_inh_spikes_train_im3 = L1_inh_spikes(:,L1_inh_spikes_train_im2_idx:L1_inh_spikes_train_im3_idx);
L1_inh_spikes_train_im4_idx = find(L1_inh_spikes(2,:)>4,1);
L1_inh_spikes_train_im4 = L1_inh_spikes(:,L1_inh_spikes_train_im3_idx:L1_inh_spikes_train_im4_idx);
L1_inh_spikes_train_im5_idx = find(L1_inh_spikes(2,:)>5,1);
L1_inh_spikes_train_im5 = L1_inh_spikes(:,L1_inh_spikes_train_im4_idx:L1_inh_spikes_train_im5_idx);
L1_inh_spikes_train_im6_idx = find(L1_inh_spikes(2,:)>6,1);
L1_inh_spikes_train_im6 = L1_inh_spikes(:,L1_inh_spikes_train_im5_idx:L1_inh_spikes_train_im6_idx);
L1_inh_spikes_train_im7_idx = find(L1_inh_spikes(2,:)>7,1);
L1_inh_spikes_train_im7 = L1_inh_spikes(:,L1_inh_spikes_train_im6_idx:L1_inh_spikes_train_im7_idx);
L1_inh_spikes_train_im8_idx = find(L1_inh_spikes(2,:)>8,1);
L1_inh_spikes_train_im8 = L1_inh_spikes(:,L1_inh_spikes_train_im7_idx:L1_inh_spikes_train_im8_idx);
L1_inh_spikes_train_im9_idx = find(L1_inh_spikes(2,:)>9,1);
L1_inh_spikes_train_im9 = L1_inh_spikes(:,L1_inh_spikes_train_im8_idx:L1_inh_spikes_train_im9_idx);
L1_inh_spikes_train_im10_idx = find(L1_inh_spikes(2,:)>10,1);
L1_inh_spikes_train_im10 = L1_inh_spikes(:,L1_inh_spikes_train_im9_idx:L1_inh_spikes_train_im10_idx);
L1_inh_spikes_train_im11_idx = find(L1_inh_spikes(2,:)>11,1);
L1_inh_spikes_train_im11 = L1_inh_spikes(:,L1_inh_spikes_train_im10_idx:L1_inh_spikes_train_im11_idx);
L1_inh_spikes_train_im12_idx = find(L1_inh_spikes(2,:)>12,1);
L1_inh_spikes_train_im12 = L1_inh_spikes(:,L1_inh_spikes_train_im11_idx:L1_inh_spikes_train_im12_idx);
L1_inh_spikes_train_im13_idx = find(L1_inh_spikes(2,:)>13,1);
L1_inh_spikes_train_im13 = L1_inh_spikes(:,L1_inh_spikes_train_im12_idx:L1_inh_spikes_train_im13_idx);
L1_inh_spikes_train_im14_idx = find(L1_inh_spikes(2,:)>14,1);
L1_inh_spikes_train_im14 = L1_inh_spikes(:,L1_inh_spikes_train_im13_idx:L1_inh_spikes_train_im14_idx);
L1_inh_spikes_train_im15_idx = find(L1_inh_spikes(2,:)>15,1);
L1_inh_spikes_train_im15 = L1_inh_spikes(:,L1_inh_spikes_train_im14_idx:L1_inh_spikes_train_im15_idx);
L1_inh_spikes_train_im16_idx = find(L1_inh_spikes(2,:)>16,1);
L1_inh_spikes_train_im16 = L1_inh_spikes(:,L1_inh_spikes_train_im15_idx:L1_inh_spikes_train_im16_idx);
 
L1_inh_spikes_test_im1_idx = find(L1_inh_spikes(2,:)>17,1);
L1_inh_spikes_test_im1 = L1_inh_spikes(:,L1_inh_spikes_train_im16_idx:L1_inh_spikes_test_im1_idx);
L1_inh_spikes_test_im2_idx = find(L1_inh_spikes(2,:)>18,1);
L1_inh_spikes_test_im2 = L1_inh_spikes(:,L1_inh_spikes_test_im1_idx:L1_inh_spikes_test_im2_idx);
L1_inh_spikes_test_im3_idx = find(L1_inh_spikes(2,:)>19,1);
L1_inh_spikes_test_im3 = L1_inh_spikes(:,L1_inh_spikes_test_im2_idx:L1_inh_spikes_test_im3_idx);
L1_inh_spikes_test_im4_idx = find(L1_inh_spikes(2,:)>20,1);
L1_inh_spikes_test_im4 = L1_inh_spikes(:,L1_inh_spikes_test_im3_idx:L1_inh_spikes_test_im4_idx);
L1_inh_spikes_test_im5_idx = find(L1_inh_spikes(2,:)>21,1);
L1_inh_spikes_test_im5 = L1_inh_spikes(:,L1_inh_spikes_test_im4_idx:L1_inh_spikes_test_im5_idx);
L1_inh_spikes_test_im6_idx = find(L1_inh_spikes(2,:)>22,1);
L1_inh_spikes_test_im6 = L1_inh_spikes(:,L1_inh_spikes_test_im5_idx:L1_inh_spikes_test_im6_idx);
L1_inh_spikes_test_im7_idx = find(L1_inh_spikes(2,:)>23,1);
L1_inh_spikes_test_im7 = L1_inh_spikes(:,L1_inh_spikes_test_im6_idx:L1_inh_spikes_test_im7_idx);
L1_inh_spikes_test_im8_idx = find(L1_inh_spikes(2,:)>24,1);
L1_inh_spikes_test_im8 = L1_inh_spikes(:,L1_inh_spikes_test_im7_idx:L1_inh_spikes_test_im8_idx);
L1_inh_spikes_test_im9_idx = find(L1_inh_spikes(2,:)>25,1);
L1_inh_spikes_test_im9 = L1_inh_spikes(:,L1_inh_spikes_test_im8_idx:L1_inh_spikes_test_im9_idx);
L1_inh_spikes_test_im10_idx = find(L1_inh_spikes(2,:)>26,1);
L1_inh_spikes_test_im10 = L1_inh_spikes(:,L1_inh_spikes_test_im9_idx:L1_inh_spikes_test_im10_idx);
L1_inh_spikes_test_im11_idx = find(L1_inh_spikes(2,:)>27,1);
L1_inh_spikes_test_im11 = L1_inh_spikes(:,L1_inh_spikes_test_im10_idx:L1_inh_spikes_test_im11_idx);
L1_inh_spikes_test_im12_idx = find(L1_inh_spikes(2,:)>28,1);
L1_inh_spikes_test_im12 = L1_inh_spikes(:,L1_inh_spikes_test_im11_idx:L1_inh_spikes_test_im12_idx);
L1_inh_spikes_test_im13_idx = find(L1_inh_spikes(2,:)>29,1);
L1_inh_spikes_test_im13 = L1_inh_spikes(:,L1_inh_spikes_test_im12_idx:L1_inh_spikes_test_im13_idx);
L1_inh_spikes_test_im14_idx = find(L1_inh_spikes(2,:)>30,1);
L1_inh_spikes_test_im14 = L1_inh_spikes(:,L1_inh_spikes_test_im13_idx:L1_inh_spikes_test_im14_idx);
L1_inh_spikes_test_im15_idx = find(L1_inh_spikes(2,:)>31,1);
L1_inh_spikes_test_im15 = L1_inh_spikes(:,L1_inh_spikes_test_im14_idx:L1_inh_spikes_test_im15_idx);
L1_inh_spikes_test_im16_idx = find(L1_inh_spikes(2,:)>32,1);
L1_inh_spikes_test_im16 = L1_inh_spikes(:,L1_inh_spikes_test_im15_idx:L1_inh_spikes_test_im16_idx);
 
L2_exc_spikes_start_idx = find(L2_exc_spikes(2,:)==0,1);
L2_exc_spikes_train_im1_idx = find(L2_exc_spikes(2,:)>1,1);
L2_exc_spikes_train_im1 = L2_exc_spikes(:,L2_exc_spikes_start_idx:L2_exc_spikes_train_im1_idx);
L2_exc_spikes_train_im2_idx = find(L2_exc_spikes(2,:)>2,1);
L2_exc_spikes_train_im2 = L2_exc_spikes(:,L2_exc_spikes_train_im1_idx:L2_exc_spikes_train_im2_idx);
L2_exc_spikes_train_im3_idx = find(L2_exc_spikes(2,:)>3,1);
L2_exc_spikes_train_im3 = L2_exc_spikes(:,L2_exc_spikes_train_im2_idx:L2_exc_spikes_train_im3_idx);
L2_exc_spikes_train_im4_idx = find(L2_exc_spikes(2,:)>4,1);
L2_exc_spikes_train_im4 = L2_exc_spikes(:,L2_exc_spikes_train_im3_idx:L2_exc_spikes_train_im4_idx);
L2_exc_spikes_train_im5_idx = find(L2_exc_spikes(2,:)>5,1);
L2_exc_spikes_train_im5 = L2_exc_spikes(:,L2_exc_spikes_train_im4_idx:L2_exc_spikes_train_im5_idx);
L2_exc_spikes_train_im6_idx = find(L2_exc_spikes(2,:)>6,1);
L2_exc_spikes_train_im6 = L2_exc_spikes(:,L2_exc_spikes_train_im5_idx:L2_exc_spikes_train_im6_idx);
L2_exc_spikes_train_im7_idx = find(L2_exc_spikes(2,:)>7,1);
L2_exc_spikes_train_im7 = L2_exc_spikes(:,L2_exc_spikes_train_im6_idx:L2_exc_spikes_train_im7_idx);
L2_exc_spikes_train_im8_idx = find(L2_exc_spikes(2,:)>8,1);
L2_exc_spikes_train_im8 = L2_exc_spikes(:,L2_exc_spikes_train_im7_idx:L2_exc_spikes_train_im8_idx);
L2_exc_spikes_train_im9_idx = find(L2_exc_spikes(2,:)>9,1);
L2_exc_spikes_train_im9 = L2_exc_spikes(:,L2_exc_spikes_train_im8_idx:L2_exc_spikes_train_im9_idx);
L2_exc_spikes_train_im10_idx = find(L2_exc_spikes(2,:)>10,1);
L2_exc_spikes_train_im10 = L2_exc_spikes(:,L2_exc_spikes_train_im9_idx:L2_exc_spikes_train_im10_idx);
L2_exc_spikes_train_im11_idx = find(L2_exc_spikes(2,:)>11,1);
L2_exc_spikes_train_im11 = L2_exc_spikes(:,L2_exc_spikes_train_im10_idx:L2_exc_spikes_train_im11_idx);
L2_exc_spikes_train_im12_idx = find(L2_exc_spikes(2,:)>12,1);
L2_exc_spikes_train_im12 = L2_exc_spikes(:,L2_exc_spikes_train_im11_idx:L2_exc_spikes_train_im12_idx);
L2_exc_spikes_train_im13_idx = find(L2_exc_spikes(2,:)>13,1);
L2_exc_spikes_train_im13 = L2_exc_spikes(:,L2_exc_spikes_train_im12_idx:L2_exc_spikes_train_im13_idx);
L2_exc_spikes_train_im14_idx = find(L2_exc_spikes(2,:)>14,1);
L2_exc_spikes_train_im14 = L2_exc_spikes(:,L2_exc_spikes_train_im13_idx:L2_exc_spikes_train_im14_idx);
L2_exc_spikes_train_im15_idx = find(L2_exc_spikes(2,:)>15,1);
L2_exc_spikes_train_im15 = L2_exc_spikes(:,L2_exc_spikes_train_im14_idx:L2_exc_spikes_train_im15_idx);
L2_exc_spikes_train_im16_idx = find(L2_exc_spikes(2,:)>16,1);
L2_exc_spikes_train_im16 = L2_exc_spikes(:,L2_exc_spikes_train_im15_idx:L2_exc_spikes_train_im16_idx);
 
L2_exc_spikes_test_im1_idx = find(L2_exc_spikes(2,:)>17,1);
L2_exc_spikes_test_im1 = L2_exc_spikes(:,L2_exc_spikes_train_im16_idx:L2_exc_spikes_test_im1_idx);
L2_exc_spikes_test_im2_idx = find(L2_exc_spikes(2,:)>18,1);
L2_exc_spikes_test_im2 = L2_exc_spikes(:,L2_exc_spikes_test_im1_idx:L2_exc_spikes_test_im2_idx);
L2_exc_spikes_test_im3_idx = find(L2_exc_spikes(2,:)>19,1);
L2_exc_spikes_test_im3 = L2_exc_spikes(:,L2_exc_spikes_test_im2_idx:L2_exc_spikes_test_im3_idx);
L2_exc_spikes_test_im4_idx = find(L2_exc_spikes(2,:)>20,1);
L2_exc_spikes_test_im4 = L2_exc_spikes(:,L2_exc_spikes_test_im3_idx:L2_exc_spikes_test_im4_idx);
L2_exc_spikes_test_im5_idx = find(L2_exc_spikes(2,:)>21,1);
L2_exc_spikes_test_im5 = L2_exc_spikes(:,L2_exc_spikes_test_im4_idx:L2_exc_spikes_test_im5_idx);
L2_exc_spikes_test_im6_idx = find(L2_exc_spikes(2,:)>22,1);
L2_exc_spikes_test_im6 = L2_exc_spikes(:,L2_exc_spikes_test_im5_idx:L2_exc_spikes_test_im6_idx);
L2_exc_spikes_test_im7_idx = find(L2_exc_spikes(2,:)>23,1);
L2_exc_spikes_test_im7 = L2_exc_spikes(:,L2_exc_spikes_test_im6_idx:L2_exc_spikes_test_im7_idx);
L2_exc_spikes_test_im8_idx = find(L2_exc_spikes(2,:)>24,1);
L2_exc_spikes_test_im8 = L2_exc_spikes(:,L2_exc_spikes_test_im7_idx:L2_exc_spikes_test_im8_idx);
L2_exc_spikes_test_im9_idx = find(L2_exc_spikes(2,:)>25,1);
L2_exc_spikes_test_im9 = L2_exc_spikes(:,L2_exc_spikes_test_im8_idx:L2_exc_spikes_test_im9_idx);
L2_exc_spikes_test_im10_idx = find(L2_exc_spikes(2,:)>26,1);
L2_exc_spikes_test_im10 = L2_exc_spikes(:,L2_exc_spikes_test_im9_idx:L2_exc_spikes_test_im10_idx);
L2_exc_spikes_test_im11_idx = find(L2_exc_spikes(2,:)>27,1);
L2_exc_spikes_test_im11 = L2_exc_spikes(:,L2_exc_spikes_test_im10_idx:L2_exc_spikes_test_im11_idx);
L2_exc_spikes_test_im12_idx = find(L2_exc_spikes(2,:)>28,1);
L2_exc_spikes_test_im12 = L2_exc_spikes(:,L2_exc_spikes_test_im11_idx:L2_exc_spikes_test_im12_idx);
L2_exc_spikes_test_im13_idx = find(L2_exc_spikes(2,:)>29,1);
L2_exc_spikes_test_im13 = L2_exc_spikes(:,L2_exc_spikes_test_im12_idx:L2_exc_spikes_test_im13_idx);
L2_exc_spikes_test_im14_idx = find(L2_exc_spikes(2,:)>30,1);
L2_exc_spikes_test_im14 = L2_exc_spikes(:,L2_exc_spikes_test_im13_idx:L2_exc_spikes_test_im14_idx);
L2_exc_spikes_test_im15_idx = find(L2_exc_spikes(2,:)>31,1);
L2_exc_spikes_test_im15 = L2_exc_spikes(:,L2_exc_spikes_test_im14_idx:L2_exc_spikes_test_im15_idx);
L2_exc_spikes_test_im16_idx = find(L2_exc_spikes(2,:)>32,1);
L2_exc_spikes_test_im16 = L2_exc_spikes(:,L2_exc_spikes_test_im15_idx:L2_exc_spikes_test_im16_idx);
 
L2_inh_spikes_start_idx = find(L2_inh_spikes(2,:)==0,1);
L2_inh_spikes_train_im1_idx = find(L2_inh_spikes(2,:)>1,1);
L2_inh_spikes_train_im1 = L2_inh_spikes(:,L2_inh_spikes_start_idx:L2_inh_spikes_train_im1_idx);
L2_inh_spikes_train_im2_idx = find(L2_inh_spikes(2,:)>2,1);
L2_inh_spikes_train_im2 = L2_inh_spikes(:,L2_inh_spikes_train_im1_idx:L2_inh_spikes_train_im2_idx);
L2_inh_spikes_train_im3_idx = find(L2_inh_spikes(2,:)>3,1);
L2_inh_spikes_train_im3 = L2_inh_spikes(:,L2_inh_spikes_train_im2_idx:L2_inh_spikes_train_im3_idx);
L2_inh_spikes_train_im4_idx = find(L2_inh_spikes(2,:)>4,1);
L2_inh_spikes_train_im4 = L2_inh_spikes(:,L2_inh_spikes_train_im3_idx:L2_inh_spikes_train_im4_idx);
L2_inh_spikes_train_im5_idx = find(L2_inh_spikes(2,:)>5,1);
L2_inh_spikes_train_im5 = L2_inh_spikes(:,L2_inh_spikes_train_im4_idx:L2_inh_spikes_train_im5_idx);
L2_inh_spikes_train_im6_idx = find(L2_inh_spikes(2,:)>6,1);
L2_inh_spikes_train_im6 = L2_inh_spikes(:,L2_inh_spikes_train_im5_idx:L2_inh_spikes_train_im6_idx);
L2_inh_spikes_train_im7_idx = find(L2_inh_spikes(2,:)>7,1);
L2_inh_spikes_train_im7 = L2_inh_spikes(:,L2_inh_spikes_train_im6_idx:L2_inh_spikes_train_im7_idx);
L2_inh_spikes_train_im8_idx = find(L2_inh_spikes(2,:)>8,1);
L2_inh_spikes_train_im8 = L2_inh_spikes(:,L2_inh_spikes_train_im7_idx:L2_inh_spikes_train_im8_idx);
L2_inh_spikes_train_im9_idx = find(L2_inh_spikes(2,:)>9,1);
L2_inh_spikes_train_im9 = L2_inh_spikes(:,L2_inh_spikes_train_im8_idx:L2_inh_spikes_train_im9_idx);
L2_inh_spikes_train_im10_idx = find(L2_inh_spikes(2,:)>10,1);
L2_inh_spikes_train_im10 = L2_inh_spikes(:,L2_inh_spikes_train_im9_idx:L2_inh_spikes_train_im10_idx);
L2_inh_spikes_train_im11_idx = find(L2_inh_spikes(2,:)>11,1);
L2_inh_spikes_train_im11 = L2_inh_spikes(:,L2_inh_spikes_train_im10_idx:L2_inh_spikes_train_im11_idx);
L2_inh_spikes_train_im12_idx = find(L2_inh_spikes(2,:)>12,1);
L2_inh_spikes_train_im12 = L2_inh_spikes(:,L2_inh_spikes_train_im11_idx:L2_inh_spikes_train_im12_idx);
L2_inh_spikes_train_im13_idx = find(L2_inh_spikes(2,:)>13,1);
L2_inh_spikes_train_im13 = L2_inh_spikes(:,L2_inh_spikes_train_im12_idx:L2_inh_spikes_train_im13_idx);
L2_inh_spikes_train_im14_idx = find(L2_inh_spikes(2,:)>14,1);
L2_inh_spikes_train_im14 = L2_inh_spikes(:,L2_inh_spikes_train_im13_idx:L2_inh_spikes_train_im14_idx);
L2_inh_spikes_train_im15_idx = find(L2_inh_spikes(2,:)>15,1);
L2_inh_spikes_train_im15 = L2_inh_spikes(:,L2_inh_spikes_train_im14_idx:L2_inh_spikes_train_im15_idx);
L2_inh_spikes_train_im16_idx = find(L2_inh_spikes(2,:)>16,1);
L2_inh_spikes_train_im16 = L2_inh_spikes(:,L2_inh_spikes_train_im15_idx:L2_inh_spikes_train_im16_idx);
 
L2_inh_spikes_test_im1_idx = find(L2_inh_spikes(2,:)>17,1);
L2_inh_spikes_test_im1 = L2_inh_spikes(:,L2_inh_spikes_train_im16_idx:L2_inh_spikes_test_im1_idx);
L2_inh_spikes_test_im2_idx = find(L2_inh_spikes(2,:)>18,1);
L2_inh_spikes_test_im2 = L2_inh_spikes(:,L2_inh_spikes_test_im1_idx:L2_inh_spikes_test_im2_idx);
L2_inh_spikes_test_im3_idx = find(L2_inh_spikes(2,:)>19,1);
L2_inh_spikes_test_im3 = L2_inh_spikes(:,L2_inh_spikes_test_im2_idx:L2_inh_spikes_test_im3_idx);
L2_inh_spikes_test_im4_idx = find(L2_inh_spikes(2,:)>20,1);
L2_inh_spikes_test_im4 = L2_inh_spikes(:,L2_inh_spikes_test_im3_idx:L2_inh_spikes_test_im4_idx);
L2_inh_spikes_test_im5_idx = find(L2_inh_spikes(2,:)>21,1);
L2_inh_spikes_test_im5 = L2_inh_spikes(:,L2_inh_spikes_test_im4_idx:L2_inh_spikes_test_im5_idx);
L2_inh_spikes_test_im6_idx = find(L2_inh_spikes(2,:)>22,1);
L2_inh_spikes_test_im6 = L2_inh_spikes(:,L2_inh_spikes_test_im5_idx:L2_inh_spikes_test_im6_idx);
L2_inh_spikes_test_im7_idx = find(L2_inh_spikes(2,:)>23,1);
L2_inh_spikes_test_im7 = L2_inh_spikes(:,L2_inh_spikes_test_im6_idx:L2_inh_spikes_test_im7_idx);
L2_inh_spikes_test_im8_idx = find(L2_inh_spikes(2,:)>24,1);
L2_inh_spikes_test_im8 = L2_inh_spikes(:,L2_inh_spikes_test_im7_idx:L2_inh_spikes_test_im8_idx);
L2_inh_spikes_test_im9_idx = find(L2_inh_spikes(2,:)>25,1);
L2_inh_spikes_test_im9 = L2_inh_spikes(:,L2_inh_spikes_test_im8_idx:L2_inh_spikes_test_im9_idx);
L2_inh_spikes_test_im10_idx = find(L2_inh_spikes(2,:)>26,1);
L2_inh_spikes_test_im10 = L2_inh_spikes(:,L2_inh_spikes_test_im9_idx:L2_inh_spikes_test_im10_idx);
L2_inh_spikes_test_im11_idx = find(L2_inh_spikes(2,:)>27,1);
L2_inh_spikes_test_im11 = L2_inh_spikes(:,L2_inh_spikes_test_im10_idx:L2_inh_spikes_test_im11_idx);
L2_inh_spikes_test_im12_idx = find(L2_inh_spikes(2,:)>28,1);
L2_inh_spikes_test_im12 = L2_inh_spikes(:,L2_inh_spikes_test_im11_idx:L2_inh_spikes_test_im12_idx);
L2_inh_spikes_test_im13_idx = find(L2_inh_spikes(2,:)>29,1);
L2_inh_spikes_test_im13 = L2_inh_spikes(:,L2_inh_spikes_test_im12_idx:L2_inh_spikes_test_im13_idx);
L2_inh_spikes_test_im14_idx = find(L2_inh_spikes(2,:)>30,1);
L2_inh_spikes_test_im14 = L2_inh_spikes(:,L2_inh_spikes_test_im13_idx:L2_inh_spikes_test_im14_idx);
L2_inh_spikes_test_im15_idx = find(L2_inh_spikes(2,:)>31,1);
L2_inh_spikes_test_im15 = L2_inh_spikes(:,L2_inh_spikes_test_im14_idx:L2_inh_spikes_test_im15_idx);
L2_inh_spikes_test_im16_idx = find(L2_inh_spikes(2,:)>32,1);
L2_inh_spikes_test_im16 = L2_inh_spikes(:,L2_inh_spikes_test_im15_idx:L2_inh_spikes_test_im16_idx);
  
L3_exc_spikes_start_idx = find(L3_exc_spikes(2,:)==0,1);
L3_exc_spikes_train_im1_idx = find(L3_exc_spikes(2,:)>1,1);
L3_exc_spikes_train_im1 = L3_exc_spikes(:,L3_exc_spikes_start_idx:L3_exc_spikes_train_im1_idx);
L3_exc_spikes_train_im2_idx = find(L3_exc_spikes(2,:)>2,1);
L3_exc_spikes_train_im2 = L3_exc_spikes(:,L3_exc_spikes_train_im1_idx:L3_exc_spikes_train_im2_idx);
L3_exc_spikes_train_im3_idx = find(L3_exc_spikes(2,:)>3,1);
L3_exc_spikes_train_im3 = L3_exc_spikes(:,L3_exc_spikes_train_im2_idx:L3_exc_spikes_train_im3_idx);
L3_exc_spikes_train_im4_idx = find(L3_exc_spikes(2,:)>4,1);
L3_exc_spikes_train_im4 = L3_exc_spikes(:,L3_exc_spikes_train_im3_idx:L3_exc_spikes_train_im4_idx);
L3_exc_spikes_train_im5_idx = find(L3_exc_spikes(2,:)>5,1);
L3_exc_spikes_train_im5 = L3_exc_spikes(:,L3_exc_spikes_train_im4_idx:L3_exc_spikes_train_im5_idx);
L3_exc_spikes_train_im6_idx = find(L3_exc_spikes(2,:)>6,1);
L3_exc_spikes_train_im6 = L3_exc_spikes(:,L3_exc_spikes_train_im5_idx:L3_exc_spikes_train_im6_idx);
L3_exc_spikes_train_im7_idx = find(L3_exc_spikes(2,:)>7,1);
L3_exc_spikes_train_im7 = L3_exc_spikes(:,L3_exc_spikes_train_im6_idx:L3_exc_spikes_train_im7_idx);
L3_exc_spikes_train_im8_idx = find(L3_exc_spikes(2,:)>8,1);
L3_exc_spikes_train_im8 = L3_exc_spikes(:,L3_exc_spikes_train_im7_idx:L3_exc_spikes_train_im8_idx);
L3_exc_spikes_train_im9_idx = find(L3_exc_spikes(2,:)>9,1);
L3_exc_spikes_train_im9 = L3_exc_spikes(:,L3_exc_spikes_train_im8_idx:L3_exc_spikes_train_im9_idx);
L3_exc_spikes_train_im10_idx = find(L3_exc_spikes(2,:)>10,1);
L3_exc_spikes_train_im10 = L3_exc_spikes(:,L3_exc_spikes_train_im9_idx:L3_exc_spikes_train_im10_idx);
L3_exc_spikes_train_im11_idx = find(L3_exc_spikes(2,:)>11,1);
L3_exc_spikes_train_im11 = L3_exc_spikes(:,L3_exc_spikes_train_im10_idx:L3_exc_spikes_train_im11_idx);
L3_exc_spikes_train_im12_idx = find(L3_exc_spikes(2,:)>12,1);
L3_exc_spikes_train_im12 = L3_exc_spikes(:,L3_exc_spikes_train_im11_idx:L3_exc_spikes_train_im12_idx);
L3_exc_spikes_train_im13_idx = find(L3_exc_spikes(2,:)>13,1);
L3_exc_spikes_train_im13 = L3_exc_spikes(:,L3_exc_spikes_train_im12_idx:L3_exc_spikes_train_im13_idx);
L3_exc_spikes_train_im14_idx = find(L3_exc_spikes(2,:)>14,1);
L3_exc_spikes_train_im14 = L3_exc_spikes(:,L3_exc_spikes_train_im13_idx:L3_exc_spikes_train_im14_idx);
L3_exc_spikes_train_im15_idx = find(L3_exc_spikes(2,:)>15,1);
L3_exc_spikes_train_im15 = L3_exc_spikes(:,L3_exc_spikes_train_im14_idx:L3_exc_spikes_train_im15_idx);
L3_exc_spikes_train_im16_idx = find(L3_exc_spikes(2,:)>16,1);
L3_exc_spikes_train_im16 = L3_exc_spikes(:,L3_exc_spikes_train_im15_idx:L3_exc_spikes_train_im16_idx);
 
L3_exc_spikes_test_im1_idx = find(L3_exc_spikes(2,:)>17,1);
L3_exc_spikes_test_im1 = L3_exc_spikes(:,L3_exc_spikes_train_im16_idx:L3_exc_spikes_test_im1_idx);
L3_exc_spikes_test_im2_idx = find(L3_exc_spikes(2,:)>18,1);
L3_exc_spikes_test_im2 = L3_exc_spikes(:,L3_exc_spikes_test_im1_idx:L3_exc_spikes_test_im2_idx);
L3_exc_spikes_test_im3_idx = find(L3_exc_spikes(2,:)>19,1);
L3_exc_spikes_test_im3 = L3_exc_spikes(:,L3_exc_spikes_test_im2_idx:L3_exc_spikes_test_im3_idx);
L3_exc_spikes_test_im4_idx = find(L3_exc_spikes(2,:)>20,1);
L3_exc_spikes_test_im4 = L3_exc_spikes(:,L3_exc_spikes_test_im3_idx:L3_exc_spikes_test_im4_idx);
L3_exc_spikes_test_im5_idx = find(L3_exc_spikes(2,:)>21,1);
L3_exc_spikes_test_im5 = L3_exc_spikes(:,L3_exc_spikes_test_im4_idx:L3_exc_spikes_test_im5_idx);
L3_exc_spikes_test_im6_idx = find(L3_exc_spikes(2,:)>22,1);
L3_exc_spikes_test_im6 = L3_exc_spikes(:,L3_exc_spikes_test_im5_idx:L3_exc_spikes_test_im6_idx);
L3_exc_spikes_test_im7_idx = find(L3_exc_spikes(2,:)>23,1);
L3_exc_spikes_test_im7 = L3_exc_spikes(:,L3_exc_spikes_test_im6_idx:L3_exc_spikes_test_im7_idx);
L3_exc_spikes_test_im8_idx = find(L3_exc_spikes(2,:)>24,1);
L3_exc_spikes_test_im8 = L3_exc_spikes(:,L3_exc_spikes_test_im7_idx:L3_exc_spikes_test_im8_idx);
L3_exc_spikes_test_im9_idx = find(L3_exc_spikes(2,:)>25,1);
L3_exc_spikes_test_im9 = L3_exc_spikes(:,L3_exc_spikes_test_im8_idx:L3_exc_spikes_test_im9_idx);
L3_exc_spikes_test_im10_idx = find(L3_exc_spikes(2,:)>26,1);
L3_exc_spikes_test_im10 = L3_exc_spikes(:,L3_exc_spikes_test_im9_idx:L3_exc_spikes_test_im10_idx);
L3_exc_spikes_test_im11_idx = find(L3_exc_spikes(2,:)>27,1);
L3_exc_spikes_test_im11 = L3_exc_spikes(:,L3_exc_spikes_test_im10_idx:L3_exc_spikes_test_im11_idx);
L3_exc_spikes_test_im12_idx = find(L3_exc_spikes(2,:)>28,1);
L3_exc_spikes_test_im12 = L3_exc_spikes(:,L3_exc_spikes_test_im11_idx:L3_exc_spikes_test_im12_idx);
L3_exc_spikes_test_im13_idx = find(L3_exc_spikes(2,:)>29,1);
L3_exc_spikes_test_im13 = L3_exc_spikes(:,L3_exc_spikes_test_im12_idx:L3_exc_spikes_test_im13_idx);
L3_exc_spikes_test_im14_idx = find(L3_exc_spikes(2,:)>30,1);
L3_exc_spikes_test_im14 = L3_exc_spikes(:,L3_exc_spikes_test_im13_idx:L3_exc_spikes_test_im14_idx);
L3_exc_spikes_test_im15_idx = find(L3_exc_spikes(2,:)>31,1);
L3_exc_spikes_test_im15 = L3_exc_spikes(:,L3_exc_spikes_test_im14_idx:L3_exc_spikes_test_im15_idx);
L3_exc_spikes_test_im16_idx = find(L3_exc_spikes(2,:)>32,1);
L3_exc_spikes_test_im16 = L3_exc_spikes(:,L3_exc_spikes_test_im15_idx:L3_exc_spikes_test_im16_idx);
 
L3_inh_spikes_start_idx = find(L3_inh_spikes(2,:)==0,1);
L3_inh_spikes_train_im1_idx = find(L3_inh_spikes(2,:)>1,1);
L3_inh_spikes_train_im1 = L3_inh_spikes(:,L3_inh_spikes_start_idx:L3_inh_spikes_train_im1_idx);
L3_inh_spikes_train_im2_idx = find(L3_inh_spikes(2,:)>2,1);
L3_inh_spikes_train_im2 = L3_inh_spikes(:,L3_inh_spikes_train_im1_idx:L3_inh_spikes_train_im2_idx);
L3_inh_spikes_train_im3_idx = find(L3_inh_spikes(2,:)>3,1);
L3_inh_spikes_train_im3 = L3_inh_spikes(:,L3_inh_spikes_train_im2_idx:L3_inh_spikes_train_im3_idx);
L3_inh_spikes_train_im4_idx = find(L3_inh_spikes(2,:)>4,1);
L3_inh_spikes_train_im4 = L3_inh_spikes(:,L3_inh_spikes_train_im3_idx:L3_inh_spikes_train_im4_idx);
L3_inh_spikes_train_im5_idx = find(L3_inh_spikes(2,:)>5,1);
L3_inh_spikes_train_im5 = L3_inh_spikes(:,L3_inh_spikes_train_im4_idx:L3_inh_spikes_train_im5_idx);
L3_inh_spikes_train_im6_idx = find(L3_inh_spikes(2,:)>6,1);
L3_inh_spikes_train_im6 = L3_inh_spikes(:,L3_inh_spikes_train_im5_idx:L3_inh_spikes_train_im6_idx);
L3_inh_spikes_train_im7_idx = find(L3_inh_spikes(2,:)>7,1);
L3_inh_spikes_train_im7 = L3_inh_spikes(:,L3_inh_spikes_train_im6_idx:L3_inh_spikes_train_im7_idx);
L3_inh_spikes_train_im8_idx = find(L3_inh_spikes(2,:)>8,1);
L3_inh_spikes_train_im8 = L3_inh_spikes(:,L3_inh_spikes_train_im7_idx:L3_inh_spikes_train_im8_idx);
L3_inh_spikes_train_im9_idx = find(L3_inh_spikes(2,:)>9,1);
L3_inh_spikes_train_im9 = L3_inh_spikes(:,L3_inh_spikes_train_im8_idx:L3_inh_spikes_train_im9_idx);
L3_inh_spikes_train_im10_idx = find(L3_inh_spikes(2,:)>10,1);
L3_inh_spikes_train_im10 = L3_inh_spikes(:,L3_inh_spikes_train_im9_idx:L3_inh_spikes_train_im10_idx);
L3_inh_spikes_train_im11_idx = find(L3_inh_spikes(2,:)>11,1);
L3_inh_spikes_train_im11 = L3_inh_spikes(:,L3_inh_spikes_train_im10_idx:L3_inh_spikes_train_im11_idx);
L3_inh_spikes_train_im12_idx = find(L3_inh_spikes(2,:)>12,1);
L3_inh_spikes_train_im12 = L3_inh_spikes(:,L3_inh_spikes_train_im11_idx:L3_inh_spikes_train_im12_idx);
L3_inh_spikes_train_im13_idx = find(L3_inh_spikes(2,:)>13,1);
L3_inh_spikes_train_im13 = L3_inh_spikes(:,L3_inh_spikes_train_im12_idx:L3_inh_spikes_train_im13_idx);
L3_inh_spikes_train_im14_idx = find(L3_inh_spikes(2,:)>14,1);
L3_inh_spikes_train_im14 = L3_inh_spikes(:,L3_inh_spikes_train_im13_idx:L3_inh_spikes_train_im14_idx);
L3_inh_spikes_train_im15_idx = find(L3_inh_spikes(2,:)>15,1);
L3_inh_spikes_train_im15 = L3_inh_spikes(:,L3_inh_spikes_train_im14_idx:L3_inh_spikes_train_im15_idx);
L3_inh_spikes_train_im16_idx = find(L3_inh_spikes(2,:)>16,1);
L3_inh_spikes_train_im16 = L3_inh_spikes(:,L3_inh_spikes_train_im15_idx:L3_inh_spikes_train_im16_idx);
 
L3_inh_spikes_test_im1_idx = find(L3_inh_spikes(2,:)>17,1);
L3_inh_spikes_test_im1 = L3_inh_spikes(:,L3_inh_spikes_train_im16_idx:L3_inh_spikes_test_im1_idx);
L3_inh_spikes_test_im2_idx = find(L3_inh_spikes(2,:)>18,1);
L3_inh_spikes_test_im2 = L3_inh_spikes(:,L3_inh_spikes_test_im1_idx:L3_inh_spikes_test_im2_idx);
L3_inh_spikes_test_im3_idx = find(L3_inh_spikes(2,:)>19,1);
L3_inh_spikes_test_im3 = L3_inh_spikes(:,L3_inh_spikes_test_im2_idx:L3_inh_spikes_test_im3_idx);
L3_inh_spikes_test_im4_idx = find(L3_inh_spikes(2,:)>20,1);
L3_inh_spikes_test_im4 = L3_inh_spikes(:,L3_inh_spikes_test_im3_idx:L3_inh_spikes_test_im4_idx);
L3_inh_spikes_test_im5_idx = find(L3_inh_spikes(2,:)>21,1);
L3_inh_spikes_test_im5 = L3_inh_spikes(:,L3_inh_spikes_test_im4_idx:L3_inh_spikes_test_im5_idx);
L3_inh_spikes_test_im6_idx = find(L3_inh_spikes(2,:)>22,1);
L3_inh_spikes_test_im6 = L3_inh_spikes(:,L3_inh_spikes_test_im5_idx:L3_inh_spikes_test_im6_idx);
L3_inh_spikes_test_im7_idx = find(L3_inh_spikes(2,:)>23,1);
L3_inh_spikes_test_im7 = L3_inh_spikes(:,L3_inh_spikes_test_im6_idx:L3_inh_spikes_test_im7_idx);
L3_inh_spikes_test_im8_idx = find(L3_inh_spikes(2,:)>24,1);
L3_inh_spikes_test_im8 = L3_inh_spikes(:,L3_inh_spikes_test_im7_idx:L3_inh_spikes_test_im8_idx);
L3_inh_spikes_test_im9_idx = find(L3_inh_spikes(2,:)>25,1);
L3_inh_spikes_test_im9 = L3_inh_spikes(:,L3_inh_spikes_test_im8_idx:L3_inh_spikes_test_im9_idx);
L3_inh_spikes_test_im10_idx = find(L3_inh_spikes(2,:)>26,1);
L3_inh_spikes_test_im10 = L3_inh_spikes(:,L3_inh_spikes_test_im9_idx:L3_inh_spikes_test_im10_idx);
L3_inh_spikes_test_im11_idx = find(L3_inh_spikes(2,:)>27,1);
L3_inh_spikes_test_im11 = L3_inh_spikes(:,L3_inh_spikes_test_im10_idx:L3_inh_spikes_test_im11_idx);
L3_inh_spikes_test_im12_idx = find(L3_inh_spikes(2,:)>28,1);
L3_inh_spikes_test_im12 = L3_inh_spikes(:,L3_inh_spikes_test_im11_idx:L3_inh_spikes_test_im12_idx);
L3_inh_spikes_test_im13_idx = find(L3_inh_spikes(2,:)>29,1);
L3_inh_spikes_test_im13 = L3_inh_spikes(:,L3_inh_spikes_test_im12_idx:L3_inh_spikes_test_im13_idx);
L3_inh_spikes_test_im14_idx = find(L3_inh_spikes(2,:)>30,1);
L3_inh_spikes_test_im14 = L3_inh_spikes(:,L3_inh_spikes_test_im13_idx:L3_inh_spikes_test_im14_idx);
L3_inh_spikes_test_im15_idx = find(L3_inh_spikes(2,:)>31,1);
L3_inh_spikes_test_im15 = L3_inh_spikes(:,L3_inh_spikes_test_im14_idx:L3_inh_spikes_test_im15_idx);
L3_inh_spikes_test_im16_idx = find(L3_inh_spikes(2,:)>32,1);
L3_inh_spikes_test_im16 = L3_inh_spikes(:,L3_inh_spikes_test_im15_idx:L3_inh_spikes_test_im16_idx);
 
 L4_exc_spikes_start_idx = find(L4_exc_spikes(2,:)==0,1);
L4_exc_spikes_train_im1_idx = find(L4_exc_spikes(2,:)>1,1);
L4_exc_spikes_train_im1 = L4_exc_spikes(:,L4_exc_spikes_start_idx:L4_exc_spikes_train_im1_idx);
L4_exc_spikes_train_im2_idx = find(L4_exc_spikes(2,:)>2,1);
L4_exc_spikes_train_im2 = L4_exc_spikes(:,L4_exc_spikes_train_im1_idx:L4_exc_spikes_train_im2_idx);
L4_exc_spikes_train_im3_idx = find(L4_exc_spikes(2,:)>3,1);
L4_exc_spikes_train_im3 = L4_exc_spikes(:,L4_exc_spikes_train_im2_idx:L4_exc_spikes_train_im3_idx);
L4_exc_spikes_train_im4_idx = find(L4_exc_spikes(2,:)>4,1);
L4_exc_spikes_train_im4 = L4_exc_spikes(:,L4_exc_spikes_train_im3_idx:L4_exc_spikes_train_im4_idx);
L4_exc_spikes_train_im5_idx = find(L4_exc_spikes(2,:)>5,1);
L4_exc_spikes_train_im5 = L4_exc_spikes(:,L4_exc_spikes_train_im4_idx:L4_exc_spikes_train_im5_idx);
L4_exc_spikes_train_im6_idx = find(L4_exc_spikes(2,:)>6,1);
L4_exc_spikes_train_im6 = L4_exc_spikes(:,L4_exc_spikes_train_im5_idx:L4_exc_spikes_train_im6_idx);
L4_exc_spikes_train_im7_idx = find(L4_exc_spikes(2,:)>7,1);
L4_exc_spikes_train_im7 = L4_exc_spikes(:,L4_exc_spikes_train_im6_idx:L4_exc_spikes_train_im7_idx);
L4_exc_spikes_train_im8_idx = find(L4_exc_spikes(2,:)>8,1);
L4_exc_spikes_train_im8 = L4_exc_spikes(:,L4_exc_spikes_train_im7_idx:L4_exc_spikes_train_im8_idx);
L4_exc_spikes_train_im9_idx = find(L4_exc_spikes(2,:)>9,1);
L4_exc_spikes_train_im9 = L4_exc_spikes(:,L4_exc_spikes_train_im8_idx:L4_exc_spikes_train_im9_idx);
L4_exc_spikes_train_im10_idx = find(L4_exc_spikes(2,:)>10,1);
L4_exc_spikes_train_im10 = L4_exc_spikes(:,L4_exc_spikes_train_im9_idx:L4_exc_spikes_train_im10_idx);
L4_exc_spikes_train_im11_idx = find(L4_exc_spikes(2,:)>11,1);
L4_exc_spikes_train_im11 = L4_exc_spikes(:,L4_exc_spikes_train_im10_idx:L4_exc_spikes_train_im11_idx);
L4_exc_spikes_train_im12_idx = find(L4_exc_spikes(2,:)>12,1);
L4_exc_spikes_train_im12 = L4_exc_spikes(:,L4_exc_spikes_train_im11_idx:L4_exc_spikes_train_im12_idx);
L4_exc_spikes_train_im13_idx = find(L4_exc_spikes(2,:)>13,1);
L4_exc_spikes_train_im13 = L4_exc_spikes(:,L4_exc_spikes_train_im12_idx:L4_exc_spikes_train_im13_idx);
L4_exc_spikes_train_im14_idx = find(L4_exc_spikes(2,:)>14,1);
L4_exc_spikes_train_im14 = L4_exc_spikes(:,L4_exc_spikes_train_im13_idx:L4_exc_spikes_train_im14_idx);
L4_exc_spikes_train_im15_idx = find(L4_exc_spikes(2,:)>15,1);
L4_exc_spikes_train_im15 = L4_exc_spikes(:,L4_exc_spikes_train_im14_idx:L4_exc_spikes_train_im15_idx);
L4_exc_spikes_train_im16_idx = find(L4_exc_spikes(2,:)>16,1);
L4_exc_spikes_train_im16 = L4_exc_spikes(:,L4_exc_spikes_train_im15_idx:L4_exc_spikes_train_im16_idx);
 
L4_exc_spikes_test_im1_idx = find(L4_exc_spikes(2,:)>17,1);
L4_exc_spikes_test_im1 = L4_exc_spikes(:,L4_exc_spikes_train_im16_idx:L4_exc_spikes_test_im1_idx);
L4_exc_spikes_test_im2_idx = find(L4_exc_spikes(2,:)>18,1);
L4_exc_spikes_test_im2 = L4_exc_spikes(:,L4_exc_spikes_test_im1_idx:L4_exc_spikes_test_im2_idx);
L4_exc_spikes_test_im3_idx = find(L4_exc_spikes(2,:)>19,1);
L4_exc_spikes_test_im3 = L4_exc_spikes(:,L4_exc_spikes_test_im2_idx:L4_exc_spikes_test_im3_idx);
L4_exc_spikes_test_im4_idx = find(L4_exc_spikes(2,:)>20,1);
L4_exc_spikes_test_im4 = L4_exc_spikes(:,L4_exc_spikes_test_im3_idx:L4_exc_spikes_test_im4_idx);
L4_exc_spikes_test_im5_idx = find(L4_exc_spikes(2,:)>21,1);
L4_exc_spikes_test_im5 = L4_exc_spikes(:,L4_exc_spikes_test_im4_idx:L4_exc_spikes_test_im5_idx);
L4_exc_spikes_test_im6_idx = find(L4_exc_spikes(2,:)>22,1);
L4_exc_spikes_test_im6 = L4_exc_spikes(:,L4_exc_spikes_test_im5_idx:L4_exc_spikes_test_im6_idx);
L4_exc_spikes_test_im7_idx = find(L4_exc_spikes(2,:)>23,1);
L4_exc_spikes_test_im7 = L4_exc_spikes(:,L4_exc_spikes_test_im6_idx:L4_exc_spikes_test_im7_idx);
L4_exc_spikes_test_im8_idx = find(L4_exc_spikes(2,:)>24,1);
L4_exc_spikes_test_im8 = L4_exc_spikes(:,L4_exc_spikes_test_im7_idx:L4_exc_spikes_test_im8_idx);
L4_exc_spikes_test_im9_idx = find(L4_exc_spikes(2,:)>25,1);
L4_exc_spikes_test_im9 = L4_exc_spikes(:,L4_exc_spikes_test_im8_idx:L4_exc_spikes_test_im9_idx);
L4_exc_spikes_test_im10_idx = find(L4_exc_spikes(2,:)>26,1);
L4_exc_spikes_test_im10 = L4_exc_spikes(:,L4_exc_spikes_test_im9_idx:L4_exc_spikes_test_im10_idx);
L4_exc_spikes_test_im11_idx = find(L4_exc_spikes(2,:)>27,1);
L4_exc_spikes_test_im11 = L4_exc_spikes(:,L4_exc_spikes_test_im10_idx:L4_exc_spikes_test_im11_idx);
L4_exc_spikes_test_im12_idx = find(L4_exc_spikes(2,:)>28,1);
L4_exc_spikes_test_im12 = L4_exc_spikes(:,L4_exc_spikes_test_im11_idx:L4_exc_spikes_test_im12_idx);
L4_exc_spikes_test_im13_idx = find(L4_exc_spikes(2,:)>29,1);
L4_exc_spikes_test_im13 = L4_exc_spikes(:,L4_exc_spikes_test_im12_idx:L4_exc_spikes_test_im13_idx);
L4_exc_spikes_test_im14_idx = find(L4_exc_spikes(2,:)>30,1);
L4_exc_spikes_test_im14 = L4_exc_spikes(:,L4_exc_spikes_test_im13_idx:L4_exc_spikes_test_im14_idx);
L4_exc_spikes_test_im15_idx = find(L4_exc_spikes(2,:)>31,1);
L4_exc_spikes_test_im15 = L4_exc_spikes(:,L4_exc_spikes_test_im14_idx:L4_exc_spikes_test_im15_idx);
L4_exc_spikes_test_im16_idx = find(L4_exc_spikes(2,:)>32,1);
L4_exc_spikes_test_im16 = L4_exc_spikes(:,L4_exc_spikes_test_im15_idx:L4_exc_spikes_test_im16_idx);
 
L4_inh_spikes_start_idx = find(L4_inh_spikes(2,:)==0,1);
L4_inh_spikes_train_im1_idx = find(L4_inh_spikes(2,:)>1,1);
L4_inh_spikes_train_im1 = L4_inh_spikes(:,L4_inh_spikes_start_idx:L4_inh_spikes_train_im1_idx);
L4_inh_spikes_train_im2_idx = find(L4_inh_spikes(2,:)>2,1);
L4_inh_spikes_train_im2 = L4_inh_spikes(:,L4_inh_spikes_train_im1_idx:L4_inh_spikes_train_im2_idx);
L4_inh_spikes_train_im3_idx = find(L4_inh_spikes(2,:)>3,1);
L4_inh_spikes_train_im3 = L4_inh_spikes(:,L4_inh_spikes_train_im2_idx:L4_inh_spikes_train_im3_idx);
L4_inh_spikes_train_im4_idx = find(L4_inh_spikes(2,:)>4,1);
L4_inh_spikes_train_im4 = L4_inh_spikes(:,L4_inh_spikes_train_im3_idx:L4_inh_spikes_train_im4_idx);
L4_inh_spikes_train_im5_idx = find(L4_inh_spikes(2,:)>5,1);
L4_inh_spikes_train_im5 = L4_inh_spikes(:,L4_inh_spikes_train_im4_idx:L4_inh_spikes_train_im5_idx);
L4_inh_spikes_train_im6_idx = find(L4_inh_spikes(2,:)>6,1);
L4_inh_spikes_train_im6 = L4_inh_spikes(:,L4_inh_spikes_train_im5_idx:L4_inh_spikes_train_im6_idx);
L4_inh_spikes_train_im7_idx = find(L4_inh_spikes(2,:)>7,1);
L4_inh_spikes_train_im7 = L4_inh_spikes(:,L4_inh_spikes_train_im6_idx:L4_inh_spikes_train_im7_idx);
L4_inh_spikes_train_im8_idx = find(L4_inh_spikes(2,:)>8,1);
L4_inh_spikes_train_im8 = L4_inh_spikes(:,L4_inh_spikes_train_im7_idx:L4_inh_spikes_train_im8_idx);
L4_inh_spikes_train_im9_idx = find(L4_inh_spikes(2,:)>9,1);
L4_inh_spikes_train_im9 = L4_inh_spikes(:,L4_inh_spikes_train_im8_idx:L4_inh_spikes_train_im9_idx);
L4_inh_spikes_train_im10_idx = find(L4_inh_spikes(2,:)>10,1);
L4_inh_spikes_train_im10 = L4_inh_spikes(:,L4_inh_spikes_train_im9_idx:L4_inh_spikes_train_im10_idx);
L4_inh_spikes_train_im11_idx = find(L4_inh_spikes(2,:)>11,1);
L4_inh_spikes_train_im11 = L4_inh_spikes(:,L4_inh_spikes_train_im10_idx:L4_inh_spikes_train_im11_idx);
L4_inh_spikes_train_im12_idx = find(L4_inh_spikes(2,:)>12,1);
L4_inh_spikes_train_im12 = L4_inh_spikes(:,L4_inh_spikes_train_im11_idx:L4_inh_spikes_train_im12_idx);
L4_inh_spikes_train_im13_idx = find(L4_inh_spikes(2,:)>13,1);
L4_inh_spikes_train_im13 = L4_inh_spikes(:,L4_inh_spikes_train_im12_idx:L4_inh_spikes_train_im13_idx);
L4_inh_spikes_train_im14_idx = find(L4_inh_spikes(2,:)>14,1);
L4_inh_spikes_train_im14 = L4_inh_spikes(:,L4_inh_spikes_train_im13_idx:L4_inh_spikes_train_im14_idx);
L4_inh_spikes_train_im15_idx = find(L4_inh_spikes(2,:)>15,1);
L4_inh_spikes_train_im15 = L4_inh_spikes(:,L4_inh_spikes_train_im14_idx:L4_inh_spikes_train_im15_idx);
L4_inh_spikes_train_im16_idx = find(L4_inh_spikes(2,:)>16,1);
L4_inh_spikes_train_im16 = L4_inh_spikes(:,L4_inh_spikes_train_im15_idx:L4_inh_spikes_train_im16_idx);
 
L4_inh_spikes_test_im1_idx = find(L4_inh_spikes(2,:)>17,1);
L4_inh_spikes_test_im1 = L4_inh_spikes(:,L4_inh_spikes_train_im16_idx:L4_inh_spikes_test_im1_idx);
L4_inh_spikes_test_im2_idx = find(L4_inh_spikes(2,:)>18,1);
L4_inh_spikes_test_im2 = L4_inh_spikes(:,L4_inh_spikes_test_im1_idx:L4_inh_spikes_test_im2_idx);
L4_inh_spikes_test_im3_idx = find(L4_inh_spikes(2,:)>19,1);
L4_inh_spikes_test_im3 = L4_inh_spikes(:,L4_inh_spikes_test_im2_idx:L4_inh_spikes_test_im3_idx);
L4_inh_spikes_test_im4_idx = find(L4_inh_spikes(2,:)>20,1);
L4_inh_spikes_test_im4 = L4_inh_spikes(:,L4_inh_spikes_test_im3_idx:L4_inh_spikes_test_im4_idx);
L4_inh_spikes_test_im5_idx = find(L4_inh_spikes(2,:)>21,1);
L4_inh_spikes_test_im5 = L4_inh_spikes(:,L4_inh_spikes_test_im4_idx:L4_inh_spikes_test_im5_idx);
L4_inh_spikes_test_im6_idx = find(L4_inh_spikes(2,:)>22,1);
L4_inh_spikes_test_im6 = L4_inh_spikes(:,L4_inh_spikes_test_im5_idx:L4_inh_spikes_test_im6_idx);
L4_inh_spikes_test_im7_idx = find(L4_inh_spikes(2,:)>23,1);
L4_inh_spikes_test_im7 = L4_inh_spikes(:,L4_inh_spikes_test_im6_idx:L4_inh_spikes_test_im7_idx);
L4_inh_spikes_test_im8_idx = find(L4_inh_spikes(2,:)>24,1);
L4_inh_spikes_test_im8 = L4_inh_spikes(:,L4_inh_spikes_test_im7_idx:L4_inh_spikes_test_im8_idx);
L4_inh_spikes_test_im9_idx = find(L4_inh_spikes(2,:)>25,1);
L4_inh_spikes_test_im9 = L4_inh_spikes(:,L4_inh_spikes_test_im8_idx:L4_inh_spikes_test_im9_idx);
L4_inh_spikes_test_im10_idx = find(L4_inh_spikes(2,:)>26,1);
L4_inh_spikes_test_im10 = L4_inh_spikes(:,L4_inh_spikes_test_im9_idx:L4_inh_spikes_test_im10_idx);
L4_inh_spikes_test_im11_idx = find(L4_inh_spikes(2,:)>27,1);
L4_inh_spikes_test_im11 = L4_inh_spikes(:,L4_inh_spikes_test_im10_idx:L4_inh_spikes_test_im11_idx);
L4_inh_spikes_test_im12_idx = find(L4_inh_spikes(2,:)>28,1);
L4_inh_spikes_test_im12 = L4_inh_spikes(:,L4_inh_spikes_test_im11_idx:L4_inh_spikes_test_im12_idx);
L4_inh_spikes_test_im13_idx = find(L4_inh_spikes(2,:)>29,1);
L4_inh_spikes_test_im13 = L4_inh_spikes(:,L4_inh_spikes_test_im12_idx:L4_inh_spikes_test_im13_idx);
L4_inh_spikes_test_im14_idx = find(L4_inh_spikes(2,:)>30,1);
L4_inh_spikes_test_im14 = L4_inh_spikes(:,L4_inh_spikes_test_im13_idx:L4_inh_spikes_test_im14_idx);
L4_inh_spikes_test_im15_idx = find(L4_inh_spikes(2,:)>31,1);
L4_inh_spikes_test_im15 = L4_inh_spikes(:,L4_inh_spikes_test_im14_idx:L4_inh_spikes_test_im15_idx);
L4_inh_spikes_test_im16_idx = find(L4_inh_spikes(2,:)>32,1);
L4_inh_spikes_test_im16 = L4_inh_spikes(:,L4_inh_spikes_test_im15_idx:L4_inh_spikes_test_im16_idx);
%% calculate average firing rates for neurons for each image
[L1e_neurons_train_im1, L1e_count_train_im1, L1e_rates_train_im1] = rates(L1_exc_spikes_train_im1,4096,1);
[L1e_neurons_train_im2, L1e_count_train_im2, L1e_rates_train_im2] = rates(L1_exc_spikes_train_im2,4096,1);
[L1e_neurons_train_im3, L1e_count_train_im3, L1e_rates_train_im3] = rates(L1_exc_spikes_train_im3,4096,1);
[L1e_neurons_train_im4, L1e_count_train_im4, L1e_rates_train_im4] = rates(L1_exc_spikes_train_im4,4096,1);
[L1e_neurons_train_im5, L1e_count_train_im5, L1e_rates_train_im5] = rates(L1_exc_spikes_train_im5,4096,1);
[L1e_neurons_train_im6, L1e_count_train_im6, L1e_rates_train_im6] = rates(L1_exc_spikes_train_im6,4096,1);
[L1e_neurons_train_im7, L1e_count_train_im7, L1e_rates_train_im7] = rates(L1_exc_spikes_train_im7,4096,1);
[L1e_neurons_train_im8, L1e_count_train_im8, L1e_rates_train_im8] = rates(L1_exc_spikes_train_im8,4096,1);
[L1e_neurons_train_im9, L1e_count_train_im9, L1e_rates_train_im9] = rates(L1_exc_spikes_train_im9,4096,1);
[L1e_neurons_train_im10, L1e_count_train_im10, L1e_rates_train_im10] = rates(L1_exc_spikes_train_im10,4096,1);
[L1e_neurons_train_im11, L1e_count_train_im11, L1e_rates_train_im11] = rates(L1_exc_spikes_train_im11,4096,1);
[L1e_neurons_train_im12, L1e_count_train_im12, L1e_rates_train_im12] = rates(L1_exc_spikes_train_im12,4096,1);
[L1e_neurons_train_im13, L1e_count_train_im13, L1e_rates_train_im13] = rates(L1_exc_spikes_train_im13,4096,1);
[L1e_neurons_train_im14, L1e_count_train_im14, L1e_rates_train_im14] = rates(L1_exc_spikes_train_im14,4096,1);
[L1e_neurons_train_im15, L1e_count_train_im15, L1e_rates_train_im15] = rates(L1_exc_spikes_train_im15,4096,1);
[L1e_neurons_train_im16, L1e_count_train_im16, L1e_rates_train_im16] = rates(L1_exc_spikes_train_im16,4096,1);
[L1e_neurons_test_im1, L1e_count_test_im1, L1e_rates_test_im1] = rates(L1_exc_spikes_test_im1,4096,1);
[L1e_neurons_test_im2, L1e_count_test_im2, L1e_rates_test_im2] = rates(L1_exc_spikes_test_im2,4096,1);
[L1e_neurons_test_im3, L1e_count_test_im3, L1e_rates_test_im3] = rates(L1_exc_spikes_test_im3,4096,1);
[L1e_neurons_test_im4, L1e_count_test_im4, L1e_rates_test_im4] = rates(L1_exc_spikes_test_im4,4096,1);
[L1e_neurons_test_im5, L1e_count_test_im5, L1e_rates_test_im5] = rates(L1_exc_spikes_test_im5,4096,1);
[L1e_neurons_test_im6, L1e_count_test_im6, L1e_rates_test_im6] = rates(L1_exc_spikes_test_im6,4096,1);
[L1e_neurons_test_im7, L1e_count_test_im7, L1e_rates_test_im7] = rates(L1_exc_spikes_test_im7,4096,1);
[L1e_neurons_test_im8, L1e_count_test_im8, L1e_rates_test_im8] = rates(L1_exc_spikes_test_im8,4096,1);
[L1e_neurons_test_im9, L1e_count_test_im9, L1e_rates_test_im9] = rates(L1_exc_spikes_test_im9,4096,1);
[L1e_neurons_test_im10, L1e_count_test_im10, L1e_rates_test_im10] = rates(L1_exc_spikes_test_im10,4096,1);
[L1e_neurons_test_im11, L1e_count_test_im11, L1e_rates_test_im11] = rates(L1_exc_spikes_test_im11,4096,1);
[L1e_neurons_test_im12, L1e_count_test_im12, L1e_rates_test_im12] = rates(L1_exc_spikes_test_im12,4096,1);
[L1e_neurons_test_im13, L1e_count_test_im13, L1e_rates_test_im13] = rates(L1_exc_spikes_test_im13,4096,1);
[L1e_neurons_test_im14, L1e_count_test_im14, L1e_rates_test_im14] = rates(L1_exc_spikes_test_im14,4096,1);
[L1e_neurons_test_im15, L1e_count_test_im15, L1e_rates_test_im15] = rates(L1_exc_spikes_test_im15,4096,1);
[L1e_neurons_test_im16, L1e_count_test_im16, L1e_rates_test_im16] = rates(L1_exc_spikes_test_im16,4096,1);
 
[L1i_neurons_train_im1, L1i_count_train_im1, L1i_rates_train_im1] = rates(L1_exc_spikes_train_im1,4096,1);
[L1i_neurons_train_im2, L1i_count_train_im2, L1i_rates_train_im2] = rates(L1_exc_spikes_train_im2,4096,1);
[L1i_neurons_train_im3, L1i_count_train_im3, L1i_rates_train_im3] = rates(L1_exc_spikes_train_im3,4096,1);
[L1i_neurons_train_im4, L1i_count_train_im4, L1i_rates_train_im4] = rates(L1_exc_spikes_train_im4,4096,1);
[L1i_neurons_train_im5, L1i_count_train_im5, L1i_rates_train_im5] = rates(L1_exc_spikes_train_im5,4096,1);
[L1i_neurons_train_im6, L1i_count_train_im6, L1i_rates_train_im6] = rates(L1_exc_spikes_train_im6,4096,1);
[L1i_neurons_train_im7, L1i_count_train_im7, L1i_rates_train_im7] = rates(L1_exc_spikes_train_im7,4096,1);
[L1i_neurons_train_im8, L1i_count_train_im8, L1i_rates_train_im8] = rates(L1_exc_spikes_train_im8,4096,1);
[L1i_neurons_train_im9, L1i_count_train_im9, L1i_rates_train_im9] = rates(L1_exc_spikes_train_im9,4096,1);
[L1i_neurons_train_im10, L1i_count_train_im10, L1i_rates_train_im10] = rates(L1_exc_spikes_train_im10,4096,1);
[L1i_neurons_train_im11, L1i_count_train_im11, L1i_rates_train_im11] = rates(L1_exc_spikes_train_im11,4096,1);
[L1i_neurons_train_im12, L1i_count_train_im12, L1i_rates_train_im12] = rates(L1_exc_spikes_train_im12,4096,1);
[L1i_neurons_train_im13, L1i_count_train_im13, L1i_rates_train_im13] = rates(L1_exc_spikes_train_im13,4096,1);
[L1i_neurons_train_im14, L1i_count_train_im14, L1i_rates_train_im14] = rates(L1_exc_spikes_train_im14,4096,1);
[L1i_neurons_train_im15, L1i_count_train_im15, L1i_rates_train_im15] = rates(L1_exc_spikes_train_im15,4096,1);
[L1i_neurons_train_im16, L1i_count_train_im16, L1i_rates_train_im16] = rates(L1_exc_spikes_train_im16,4096,1);
[L1i_neurons_test_im1, L1i_count_test_im1, L1i_rates_test_im1] = rates(L1_exc_spikes_test_im1,4096,1);
[L1i_neurons_test_im2, L1i_count_test_im2, L1i_rates_test_im2] = rates(L1_exc_spikes_test_im2,4096,1);
[L1i_neurons_test_im3, L1i_count_test_im3, L1i_rates_test_im3] = rates(L1_exc_spikes_test_im3,4096,1);
[L1i_neurons_test_im4, L1i_count_test_im4, L1i_rates_test_im4] = rates(L1_exc_spikes_test_im4,4096,1);
[L1i_neurons_test_im5, L1i_count_test_im5, L1i_rates_test_im5] = rates(L1_exc_spikes_test_im5,4096,1);
[L1i_neurons_test_im6, L1i_count_test_im6, L1i_rates_test_im6] = rates(L1_exc_spikes_test_im6,4096,1);
[L1i_neurons_test_im7, L1i_count_test_im7, L1i_rates_test_im7] = rates(L1_exc_spikes_test_im7,4096,1);
[L1i_neurons_test_im8, L1i_count_test_im8, L1i_rates_test_im8] = rates(L1_exc_spikes_test_im8,4096,1);
[L1i_neurons_test_im9, L1i_count_test_im9, L1i_rates_test_im9] = rates(L1_exc_spikes_test_im9,4096,1);
[L1i_neurons_test_im10, L1i_count_test_im10, L1i_rates_test_im10] = rates(L1_exc_spikes_test_im10,4096,1);
[L1i_neurons_test_im11, L1i_count_test_im11, L1i_rates_test_im11] = rates(L1_exc_spikes_test_im11,4096,1);
[L1i_neurons_test_im12, L1i_count_test_im12, L1i_rates_test_im12] = rates(L1_exc_spikes_test_im12,4096,1);
[L1i_neurons_test_im13, L1i_count_test_im13, L1i_rates_test_im13] = rates(L1_exc_spikes_test_im13,4096,1);
[L1i_neurons_test_im14, L1i_count_test_im14, L1i_rates_test_im14] = rates(L1_exc_spikes_test_im14,4096,1);
[L1i_neurons_test_im15, L1i_count_test_im15, L1i_rates_test_im15] = rates(L1_exc_spikes_test_im15,4096,1);
[L1i_neurons_test_im16, L1i_count_test_im16, L1i_rates_test_im16] = rates(L1_exc_spikes_test_im16,4096,1);
 
[L2e_neurons_train_im1, L2e_count_train_im1, L2e_rates_train_im1] = rates(L2_exc_spikes_train_im1,4096,1);
[L2e_neurons_train_im2, L2e_count_train_im2, L2e_rates_train_im2] = rates(L2_exc_spikes_train_im2,4096,1);
[L2e_neurons_train_im3, L2e_count_train_im3, L2e_rates_train_im3] = rates(L2_exc_spikes_train_im3,4096,1);
[L2e_neurons_train_im4, L2e_count_train_im4, L2e_rates_train_im4] = rates(L2_exc_spikes_train_im4,4096,1);
[L2e_neurons_train_im5, L2e_count_train_im5, L2e_rates_train_im5] = rates(L2_exc_spikes_train_im5,4096,1);
[L2e_neurons_train_im6, L2e_count_train_im6, L2e_rates_train_im6] = rates(L2_exc_spikes_train_im6,4096,1);
[L2e_neurons_train_im7, L2e_count_train_im7, L2e_rates_train_im7] = rates(L2_exc_spikes_train_im7,4096,1);
[L2e_neurons_train_im8, L2e_count_train_im8, L2e_rates_train_im8] = rates(L2_exc_spikes_train_im8,4096,1);
[L2e_neurons_train_im9, L2e_count_train_im9, L2e_rates_train_im9] = rates(L2_exc_spikes_train_im9,4096,1);
[L2e_neurons_train_im10, L2e_count_train_im10, L2e_rates_train_im10] = rates(L2_exc_spikes_train_im10,4096,1);
[L2e_neurons_train_im11, L2e_count_train_im11, L2e_rates_train_im11] = rates(L2_exc_spikes_train_im11,4096,1);
[L2e_neurons_train_im12, L2e_count_train_im12, L2e_rates_train_im12] = rates(L2_exc_spikes_train_im12,4096,1);
[L2e_neurons_train_im13, L2e_count_train_im13, L2e_rates_train_im13] = rates(L2_exc_spikes_train_im13,4096,1);
[L2e_neurons_train_im14, L2e_count_train_im14, L2e_rates_train_im14] = rates(L2_exc_spikes_train_im14,4096,1);
[L2e_neurons_train_im15, L2e_count_train_im15, L2e_rates_train_im15] = rates(L2_exc_spikes_train_im15,4096,1);
[L2e_neurons_train_im16, L2e_count_train_im16, L2e_rates_train_im16] = rates(L2_exc_spikes_train_im16,4096,1);
[L2e_neurons_test_im1, L2e_count_test_im1, L2e_rates_test_im1] = rates(L2_exc_spikes_test_im1,4096,1);
[L2e_neurons_test_im2, L2e_count_test_im2, L2e_rates_test_im2] = rates(L2_exc_spikes_test_im2,4096,1);
[L2e_neurons_test_im3, L2e_count_test_im3, L2e_rates_test_im3] = rates(L2_exc_spikes_test_im3,4096,1);
[L2e_neurons_test_im4, L2e_count_test_im4, L2e_rates_test_im4] = rates(L2_exc_spikes_test_im4,4096,1);
[L2e_neurons_test_im5, L2e_count_test_im5, L2e_rates_test_im5] = rates(L2_exc_spikes_test_im5,4096,1);
[L2e_neurons_test_im6, L2e_count_test_im6, L2e_rates_test_im6] = rates(L2_exc_spikes_test_im6,4096,1);
[L2e_neurons_test_im7, L2e_count_test_im7, L2e_rates_test_im7] = rates(L2_exc_spikes_test_im7,4096,1);
[L2e_neurons_test_im8, L2e_count_test_im8, L2e_rates_test_im8] = rates(L2_exc_spikes_test_im8,4096,1);
[L2e_neurons_test_im9, L2e_count_test_im9, L2e_rates_test_im9] = rates(L2_exc_spikes_test_im9,4096,1);
[L2e_neurons_test_im10, L2e_count_test_im10, L2e_rates_test_im10] = rates(L2_exc_spikes_test_im10,4096,1);
[L2e_neurons_test_im11, L2e_count_test_im11, L2e_rates_test_im11] = rates(L2_exc_spikes_test_im11,4096,1);
[L2e_neurons_test_im12, L2e_count_test_im12, L2e_rates_test_im12] = rates(L2_exc_spikes_test_im12,4096,1);
[L2e_neurons_test_im13, L2e_count_test_im13, L2e_rates_test_im13] = rates(L2_exc_spikes_test_im13,4096,1);
[L2e_neurons_test_im14, L2e_count_test_im14, L2e_rates_test_im14] = rates(L2_exc_spikes_test_im14,4096,1);
[L2e_neurons_test_im15, L2e_count_test_im15, L2e_rates_test_im15] = rates(L2_exc_spikes_test_im15,4096,1);
[L2e_neurons_test_im16, L2e_count_test_im16, L2e_rates_test_im16] = rates(L2_exc_spikes_test_im16,4096,1);
 
[L2i_neurons_train_im1, L2i_count_train_im1, L2i_rates_train_im1] = rates(L2_exc_spikes_train_im1,4096,1);
[L2i_neurons_train_im2, L2i_count_train_im2, L2i_rates_train_im2] = rates(L2_exc_spikes_train_im2,4096,1);
[L2i_neurons_train_im3, L2i_count_train_im3, L2i_rates_train_im3] = rates(L2_exc_spikes_train_im3,4096,1);
[L2i_neurons_train_im4, L2i_count_train_im4, L2i_rates_train_im4] = rates(L2_exc_spikes_train_im4,4096,1);
[L2i_neurons_train_im5, L2i_count_train_im5, L2i_rates_train_im5] = rates(L2_exc_spikes_train_im5,4096,1);
[L2i_neurons_train_im6, L2i_count_train_im6, L2i_rates_train_im6] = rates(L2_exc_spikes_train_im6,4096,1);
[L2i_neurons_train_im7, L2i_count_train_im7, L2i_rates_train_im7] = rates(L2_exc_spikes_train_im7,4096,1);
[L2i_neurons_train_im8, L2i_count_train_im8, L2i_rates_train_im8] = rates(L2_exc_spikes_train_im8,4096,1);
[L2i_neurons_train_im9, L2i_count_train_im9, L2i_rates_train_im9] = rates(L2_exc_spikes_train_im9,4096,1);
[L2i_neurons_train_im10, L2i_count_train_im10, L2i_rates_train_im10] = rates(L2_exc_spikes_train_im10,4096,1);
[L2i_neurons_train_im11, L2i_count_train_im11, L2i_rates_train_im11] = rates(L2_exc_spikes_train_im11,4096,1);
[L2i_neurons_train_im12, L2i_count_train_im12, L2i_rates_train_im12] = rates(L2_exc_spikes_train_im12,4096,1);
[L2i_neurons_train_im13, L2i_count_train_im13, L2i_rates_train_im13] = rates(L2_exc_spikes_train_im13,4096,1);
[L2i_neurons_train_im14, L2i_count_train_im14, L2i_rates_train_im14] = rates(L2_exc_spikes_train_im14,4096,1);
[L2i_neurons_train_im15, L2i_count_train_im15, L2i_rates_train_im15] = rates(L2_exc_spikes_train_im15,4096,1);
[L2i_neurons_train_im16, L2i_count_train_im16, L2i_rates_train_im16] = rates(L2_exc_spikes_train_im16,4096,1);
[L2i_neurons_test_im1, L2i_count_test_im1, L2i_rates_test_im1] = rates(L2_exc_spikes_test_im1,4096,1);
[L2i_neurons_test_im2, L2i_count_test_im2, L2i_rates_test_im2] = rates(L2_exc_spikes_test_im2,4096,1);
[L2i_neurons_test_im3, L2i_count_test_im3, L2i_rates_test_im3] = rates(L2_exc_spikes_test_im3,4096,1);
[L2i_neurons_test_im4, L2i_count_test_im4, L2i_rates_test_im4] = rates(L2_exc_spikes_test_im4,4096,1);
[L2i_neurons_test_im5, L2i_count_test_im5, L2i_rates_test_im5] = rates(L2_exc_spikes_test_im5,4096,1);
[L2i_neurons_test_im6, L2i_count_test_im6, L2i_rates_test_im6] = rates(L2_exc_spikes_test_im6,4096,1);
[L2i_neurons_test_im7, L2i_count_test_im7, L2i_rates_test_im7] = rates(L2_exc_spikes_test_im7,4096,1);
[L2i_neurons_test_im8, L2i_count_test_im8, L2i_rates_test_im8] = rates(L2_exc_spikes_test_im8,4096,1);
[L2i_neurons_test_im9, L2i_count_test_im9, L2i_rates_test_im9] = rates(L2_exc_spikes_test_im9,4096,1);
[L2i_neurons_test_im10, L2i_count_test_im10, L2i_rates_test_im10] = rates(L2_exc_spikes_test_im10,4096,1);
[L2i_neurons_test_im11, L2i_count_test_im11, L2i_rates_test_im11] = rates(L2_exc_spikes_test_im11,4096,1);
[L2i_neurons_test_im12, L2i_count_test_im12, L2i_rates_test_im12] = rates(L2_exc_spikes_test_im12,4096,1);
[L2i_neurons_test_im13, L2i_count_test_im13, L2i_rates_test_im13] = rates(L2_exc_spikes_test_im13,4096,1);
[L2i_neurons_test_im14, L2i_count_test_im14, L2i_rates_test_im14] = rates(L2_exc_spikes_test_im14,4096,1);
[L2i_neurons_test_im15, L2i_count_test_im15, L2i_rates_test_im15] = rates(L2_exc_spikes_test_im15,4096,1);
[L2i_neurons_test_im16, L2i_count_test_im16, L2i_rates_test_im16] = rates(L2_exc_spikes_test_im16,4096,1); 
 
[L3e_neurons_train_im1, L3e_count_train_im1, L3e_rates_train_im1] = rates(L3_exc_spikes_train_im1,4096,1);
[L3e_neurons_train_im2, L3e_count_train_im2, L3e_rates_train_im2] = rates(L3_exc_spikes_train_im2,4096,1);
[L3e_neurons_train_im3, L3e_count_train_im3, L3e_rates_train_im3] = rates(L3_exc_spikes_train_im3,4096,1);
[L3e_neurons_train_im4, L3e_count_train_im4, L3e_rates_train_im4] = rates(L3_exc_spikes_train_im4,4096,1);
[L3e_neurons_train_im5, L3e_count_train_im5, L3e_rates_train_im5] = rates(L3_exc_spikes_train_im5,4096,1);
[L3e_neurons_train_im6, L3e_count_train_im6, L3e_rates_train_im6] = rates(L3_exc_spikes_train_im6,4096,1);
[L3e_neurons_train_im7, L3e_count_train_im7, L3e_rates_train_im7] = rates(L3_exc_spikes_train_im7,4096,1);
[L3e_neurons_train_im8, L3e_count_train_im8, L3e_rates_train_im8] = rates(L3_exc_spikes_train_im8,4096,1);
[L3e_neurons_train_im9, L3e_count_train_im9, L3e_rates_train_im9] = rates(L3_exc_spikes_train_im9,4096,1);
[L3e_neurons_train_im10, L3e_count_train_im10, L3e_rates_train_im10] = rates(L3_exc_spikes_train_im10,4096,1);
[L3e_neurons_train_im11, L3e_count_train_im11, L3e_rates_train_im11] = rates(L3_exc_spikes_train_im11,4096,1);
[L3e_neurons_train_im12, L3e_count_train_im12, L3e_rates_train_im12] = rates(L3_exc_spikes_train_im12,4096,1);
[L3e_neurons_train_im13, L3e_count_train_im13, L3e_rates_train_im13] = rates(L3_exc_spikes_train_im13,4096,1);
[L3e_neurons_train_im14, L3e_count_train_im14, L3e_rates_train_im14] = rates(L3_exc_spikes_train_im14,4096,1);
[L3e_neurons_train_im15, L3e_count_train_im15, L3e_rates_train_im15] = rates(L3_exc_spikes_train_im15,4096,1);
[L3e_neurons_train_im16, L3e_count_train_im16, L3e_rates_train_im16] = rates(L3_exc_spikes_train_im16,4096,1);
[L3e_neurons_test_im1, L3e_count_test_im1, L3e_rates_test_im1] = rates(L3_exc_spikes_test_im1,4096,1);
[L3e_neurons_test_im2, L3e_count_test_im2, L3e_rates_test_im2] = rates(L3_exc_spikes_test_im2,4096,1);
[L3e_neurons_test_im3, L3e_count_test_im3, L3e_rates_test_im3] = rates(L3_exc_spikes_test_im3,4096,1);
[L3e_neurons_test_im4, L3e_count_test_im4, L3e_rates_test_im4] = rates(L3_exc_spikes_test_im4,4096,1);
[L3e_neurons_test_im5, L3e_count_test_im5, L3e_rates_test_im5] = rates(L3_exc_spikes_test_im5,4096,1);
[L3e_neurons_test_im6, L3e_count_test_im6, L3e_rates_test_im6] = rates(L3_exc_spikes_test_im6,4096,1);
[L3e_neurons_test_im7, L3e_count_test_im7, L3e_rates_test_im7] = rates(L3_exc_spikes_test_im7,4096,1);
[L3e_neurons_test_im8, L3e_count_test_im8, L3e_rates_test_im8] = rates(L3_exc_spikes_test_im8,4096,1);
[L3e_neurons_test_im9, L3e_count_test_im9, L3e_rates_test_im9] = rates(L3_exc_spikes_test_im9,4096,1);
[L3e_neurons_test_im10, L3e_count_test_im10, L3e_rates_test_im10] = rates(L3_exc_spikes_test_im10,4096,1);
[L3e_neurons_test_im11, L3e_count_test_im11, L3e_rates_test_im11] = rates(L3_exc_spikes_test_im11,4096,1);
[L3e_neurons_test_im12, L3e_count_test_im12, L3e_rates_test_im12] = rates(L3_exc_spikes_test_im12,4096,1);
[L3e_neurons_test_im13, L3e_count_test_im13, L3e_rates_test_im13] = rates(L3_exc_spikes_test_im13,4096,1);
[L3e_neurons_test_im14, L3e_count_test_im14, L3e_rates_test_im14] = rates(L3_exc_spikes_test_im14,4096,1);
[L3e_neurons_test_im15, L3e_count_test_im15, L3e_rates_test_im15] = rates(L3_exc_spikes_test_im15,4096,1);
[L3e_neurons_test_im16, L3e_count_test_im16, L3e_rates_test_im16] = rates(L3_exc_spikes_test_im16,4096,1);
 
[L3i_neurons_train_im1, L3i_count_train_im1, L3i_rates_train_im1] = rates(L3_exc_spikes_train_im1,4096,1);
[L3i_neurons_train_im2, L3i_count_train_im2, L3i_rates_train_im2] = rates(L3_exc_spikes_train_im2,4096,1);
[L3i_neurons_train_im3, L3i_count_train_im3, L3i_rates_train_im3] = rates(L3_exc_spikes_train_im3,4096,1);
[L3i_neurons_train_im4, L3i_count_train_im4, L3i_rates_train_im4] = rates(L3_exc_spikes_train_im4,4096,1);
[L3i_neurons_train_im5, L3i_count_train_im5, L3i_rates_train_im5] = rates(L3_exc_spikes_train_im5,4096,1);
[L3i_neurons_train_im6, L3i_count_train_im6, L3i_rates_train_im6] = rates(L3_exc_spikes_train_im6,4096,1);
[L3i_neurons_train_im7, L3i_count_train_im7, L3i_rates_train_im7] = rates(L3_exc_spikes_train_im7,4096,1);
[L3i_neurons_train_im8, L3i_count_train_im8, L3i_rates_train_im8] = rates(L3_exc_spikes_train_im8,4096,1);
[L3i_neurons_train_im9, L3i_count_train_im9, L3i_rates_train_im9] = rates(L3_exc_spikes_train_im9,4096,1);
[L3i_neurons_train_im10, L3i_count_train_im10, L3i_rates_train_im10] = rates(L3_exc_spikes_train_im10,4096,1);
[L3i_neurons_train_im11, L3i_count_train_im11, L3i_rates_train_im11] = rates(L3_exc_spikes_train_im11,4096,1);
[L3i_neurons_train_im12, L3i_count_train_im12, L3i_rates_train_im12] = rates(L3_exc_spikes_train_im12,4096,1);
[L3i_neurons_train_im13, L3i_count_train_im13, L3i_rates_train_im13] = rates(L3_exc_spikes_train_im13,4096,1);
[L3i_neurons_train_im14, L3i_count_train_im14, L3i_rates_train_im14] = rates(L3_exc_spikes_train_im14,4096,1);
[L3i_neurons_train_im15, L3i_count_train_im15, L3i_rates_train_im15] = rates(L3_exc_spikes_train_im15,4096,1);
[L3i_neurons_train_im16, L3i_count_train_im16, L3i_rates_train_im16] = rates(L3_exc_spikes_train_im16,4096,1);
[L3i_neurons_test_im1, L3i_count_test_im1, L3i_rates_test_im1] = rates(L3_exc_spikes_test_im1,4096,1);
[L3i_neurons_test_im2, L3i_count_test_im2, L3i_rates_test_im2] = rates(L3_exc_spikes_test_im2,4096,1);
[L3i_neurons_test_im3, L3i_count_test_im3, L3i_rates_test_im3] = rates(L3_exc_spikes_test_im3,4096,1);
[L3i_neurons_test_im4, L3i_count_test_im4, L3i_rates_test_im4] = rates(L3_exc_spikes_test_im4,4096,1);
[L3i_neurons_test_im5, L3i_count_test_im5, L3i_rates_test_im5] = rates(L3_exc_spikes_test_im5,4096,1);
[L3i_neurons_test_im6, L3i_count_test_im6, L3i_rates_test_im6] = rates(L3_exc_spikes_test_im6,4096,1);
[L3i_neurons_test_im7, L3i_count_test_im7, L3i_rates_test_im7] = rates(L3_exc_spikes_test_im7,4096,1);
[L3i_neurons_test_im8, L3i_count_test_im8, L3i_rates_test_im8] = rates(L3_exc_spikes_test_im8,4096,1);
[L3i_neurons_test_im9, L3i_count_test_im9, L3i_rates_test_im9] = rates(L3_exc_spikes_test_im9,4096,1);
[L3i_neurons_test_im10, L3i_count_test_im10, L3i_rates_test_im10] = rates(L3_exc_spikes_test_im10,4096,1);
[L3i_neurons_test_im11, L3i_count_test_im11, L3i_rates_test_im11] = rates(L3_exc_spikes_test_im11,4096,1);
[L3i_neurons_test_im12, L3i_count_test_im12, L3i_rates_test_im12] = rates(L3_exc_spikes_test_im12,4096,1);
[L3i_neurons_test_im13, L3i_count_test_im13, L3i_rates_test_im13] = rates(L3_exc_spikes_test_im13,4096,1);
[L3i_neurons_test_im14, L3i_count_test_im14, L3i_rates_test_im14] = rates(L3_exc_spikes_test_im14,4096,1);
[L3i_neurons_test_im15, L3i_count_test_im15, L3i_rates_test_im15] = rates(L3_exc_spikes_test_im15,4096,1);
[L3i_neurons_test_im16, L3i_count_test_im16, L3i_rates_test_im16] = rates(L3_exc_spikes_test_im16,4096,1);
 
 [L4e_neurons_train_im1, L4e_count_train_im1, L4e_rates_train_im1] = rates(L4_exc_spikes_train_im1,4096,1);
[L4e_neurons_train_im2, L4e_count_train_im2, L4e_rates_train_im2] = rates(L4_exc_spikes_train_im2,4096,1);
[L4e_neurons_train_im3, L4e_count_train_im3, L4e_rates_train_im3] = rates(L4_exc_spikes_train_im3,4096,1);
[L4e_neurons_train_im4, L4e_count_train_im4, L4e_rates_train_im4] = rates(L4_exc_spikes_train_im4,4096,1);
[L4e_neurons_train_im5, L4e_count_train_im5, L4e_rates_train_im5] = rates(L4_exc_spikes_train_im5,4096,1);
[L4e_neurons_train_im6, L4e_count_train_im6, L4e_rates_train_im6] = rates(L4_exc_spikes_train_im6,4096,1);
[L4e_neurons_train_im7, L4e_count_train_im7, L4e_rates_train_im7] = rates(L4_exc_spikes_train_im7,4096,1);
[L4e_neurons_train_im8, L4e_count_train_im8, L4e_rates_train_im8] = rates(L4_exc_spikes_train_im8,4096,1);
[L4e_neurons_train_im9, L4e_count_train_im9, L4e_rates_train_im9] = rates(L4_exc_spikes_train_im9,4096,1);
[L4e_neurons_train_im10, L4e_count_train_im10, L4e_rates_train_im10] = rates(L4_exc_spikes_train_im10,4096,1);
[L4e_neurons_train_im11, L4e_count_train_im11, L4e_rates_train_im11] = rates(L4_exc_spikes_train_im11,4096,1);
[L4e_neurons_train_im12, L4e_count_train_im12, L4e_rates_train_im12] = rates(L4_exc_spikes_train_im12,4096,1);
[L4e_neurons_train_im13, L4e_count_train_im13, L4e_rates_train_im13] = rates(L4_exc_spikes_train_im13,4096,1);
[L4e_neurons_train_im14, L4e_count_train_im14, L4e_rates_train_im14] = rates(L4_exc_spikes_train_im14,4096,1);
[L4e_neurons_train_im15, L4e_count_train_im15, L4e_rates_train_im15] = rates(L4_exc_spikes_train_im15,4096,1);
[L4e_neurons_train_im16, L4e_count_train_im16, L4e_rates_train_im16] = rates(L4_exc_spikes_train_im16,4096,1);
[L4e_neurons_test_im1, L4e_count_test_im1, L4e_rates_test_im1] = rates(L4_exc_spikes_test_im1,4096,1);
[L4e_neurons_test_im2, L4e_count_test_im2, L4e_rates_test_im2] = rates(L4_exc_spikes_test_im2,4096,1);
[L4e_neurons_test_im3, L4e_count_test_im3, L4e_rates_test_im3] = rates(L4_exc_spikes_test_im3,4096,1);
[L4e_neurons_test_im4, L4e_count_test_im4, L4e_rates_test_im4] = rates(L4_exc_spikes_test_im4,4096,1);
[L4e_neurons_test_im5, L4e_count_test_im5, L4e_rates_test_im5] = rates(L4_exc_spikes_test_im5,4096,1);
[L4e_neurons_test_im6, L4e_count_test_im6, L4e_rates_test_im6] = rates(L4_exc_spikes_test_im6,4096,1);
[L4e_neurons_test_im7, L4e_count_test_im7, L4e_rates_test_im7] = rates(L4_exc_spikes_test_im7,4096,1);
[L4e_neurons_test_im8, L4e_count_test_im8, L4e_rates_test_im8] = rates(L4_exc_spikes_test_im8,4096,1);
[L4e_neurons_test_im9, L4e_count_test_im9, L4e_rates_test_im9] = rates(L4_exc_spikes_test_im9,4096,1);
[L4e_neurons_test_im10, L4e_count_test_im10, L4e_rates_test_im10] = rates(L4_exc_spikes_test_im10,4096,1);
[L4e_neurons_test_im11, L4e_count_test_im11, L4e_rates_test_im11] = rates(L4_exc_spikes_test_im11,4096,1);
[L4e_neurons_test_im12, L4e_count_test_im12, L4e_rates_test_im12] = rates(L4_exc_spikes_test_im12,4096,1);
[L4e_neurons_test_im13, L4e_count_test_im13, L4e_rates_test_im13] = rates(L4_exc_spikes_test_im13,4096,1);
[L4e_neurons_test_im14, L4e_count_test_im14, L4e_rates_test_im14] = rates(L4_exc_spikes_test_im14,4096,1);
[L4e_neurons_test_im15, L4e_count_test_im15, L4e_rates_test_im15] = rates(L4_exc_spikes_test_im15,4096,1);
[L4e_neurons_test_im16, L4e_count_test_im16, L4e_rates_test_im16] = rates(L4_exc_spikes_test_im16,4096,1);
 
[L4i_neurons_train_im1, L4i_count_train_im1, L4i_rates_train_im1] = rates(L4_exc_spikes_train_im1,4096,1);
[L4i_neurons_train_im2, L4i_count_train_im2, L4i_rates_train_im2] = rates(L4_exc_spikes_train_im2,4096,1);
[L4i_neurons_train_im3, L4i_count_train_im3, L4i_rates_train_im3] = rates(L4_exc_spikes_train_im3,4096,1);
[L4i_neurons_train_im4, L4i_count_train_im4, L4i_rates_train_im4] = rates(L4_exc_spikes_train_im4,4096,1);
[L4i_neurons_train_im5, L4i_count_train_im5, L4i_rates_train_im5] = rates(L4_exc_spikes_train_im5,4096,1);
[L4i_neurons_train_im6, L4i_count_train_im6, L4i_rates_train_im6] = rates(L4_exc_spikes_train_im6,4096,1);
[L4i_neurons_train_im7, L4i_count_train_im7, L4i_rates_train_im7] = rates(L4_exc_spikes_train_im7,4096,1);
[L4i_neurons_train_im8, L4i_count_train_im8, L4i_rates_train_im8] = rates(L4_exc_spikes_train_im8,4096,1);
[L4i_neurons_train_im9, L4i_count_train_im9, L4i_rates_train_im9] = rates(L4_exc_spikes_train_im9,4096,1);
[L4i_neurons_train_im10, L4i_count_train_im10, L4i_rates_train_im10] = rates(L4_exc_spikes_train_im10,4096,1);
[L4i_neurons_train_im11, L4i_count_train_im11, L4i_rates_train_im11] = rates(L4_exc_spikes_train_im11,4096,1);
[L4i_neurons_train_im12, L4i_count_train_im12, L4i_rates_train_im12] = rates(L4_exc_spikes_train_im12,4096,1);
[L4i_neurons_train_im13, L4i_count_train_im13, L4i_rates_train_im13] = rates(L4_exc_spikes_train_im13,4096,1);
[L4i_neurons_train_im14, L4i_count_train_im14, L4i_rates_train_im14] = rates(L4_exc_spikes_train_im14,4096,1);
[L4i_neurons_train_im15, L4i_count_train_im15, L4i_rates_train_im15] = rates(L4_exc_spikes_train_im15,4096,1);
[L4i_neurons_train_im16, L4i_count_train_im16, L4i_rates_train_im16] = rates(L4_exc_spikes_train_im16,4096,1);
[L4i_neurons_test_im1, L4i_count_test_im1, L4i_rates_test_im1] = rates(L4_exc_spikes_test_im1,4096,1);
[L4i_neurons_test_im2, L4i_count_test_im2, L4i_rates_test_im2] = rates(L4_exc_spikes_test_im2,4096,1);
[L4i_neurons_test_im3, L4i_count_test_im3, L4i_rates_test_im3] = rates(L4_exc_spikes_test_im3,4096,1);
[L4i_neurons_test_im4, L4i_count_test_im4, L4i_rates_test_im4] = rates(L4_exc_spikes_test_im4,4096,1);
[L4i_neurons_test_im5, L4i_count_test_im5, L4i_rates_test_im5] = rates(L4_exc_spikes_test_im5,4096,1);
[L4i_neurons_test_im6, L4i_count_test_im6, L4i_rates_test_im6] = rates(L4_exc_spikes_test_im6,4096,1);
[L4i_neurons_test_im7, L4i_count_test_im7, L4i_rates_test_im7] = rates(L4_exc_spikes_test_im7,4096,1);
[L4i_neurons_test_im8, L4i_count_test_im8, L4i_rates_test_im8] = rates(L4_exc_spikes_test_im8,4096,1);
[L4i_neurons_test_im9, L4i_count_test_im9, L4i_rates_test_im9] = rates(L4_exc_spikes_test_im9,4096,1);
[L4i_neurons_test_im10, L4i_count_test_im10, L4i_rates_test_im10] = rates(L4_exc_spikes_test_im10,4096,1);
[L4i_neurons_test_im11, L4i_count_test_im11, L4i_rates_test_im11] = rates(L4_exc_spikes_test_im11,4096,1);
[L4i_neurons_test_im12, L4i_count_test_im12, L4i_rates_test_im12] = rates(L4_exc_spikes_test_im12,4096,1);
[L4i_neurons_test_im13, L4i_count_test_im13, L4i_rates_test_im13] = rates(L4_exc_spikes_test_im13,4096,1);
[L4i_neurons_test_im14, L4i_count_test_im14, L4i_rates_test_im14] = rates(L4_exc_spikes_test_im14,4096,1);
[L4i_neurons_test_im15, L4i_count_test_im15, L4i_rates_test_im15] = rates(L4_exc_spikes_test_im15,4096,1);
[L4i_neurons_test_im16, L4i_count_test_im16, L4i_rates_test_im16] = rates(L4_exc_spikes_test_im16,4096,1);
%% plot average firing rates
 
[L3e_neurons, L3e_count, L3e_rates] = rates(L3_exc_spikes(:,L3_exc_spikes_train_im16_idx:end),4096,16);
[L4e_neurons, L4e_count, L4e_rates] = rates(L4_exc_spikes(:,L4_exc_spikes_train_im16_idx:end),4096,16);
 
figure('Renderer', 'painters', 'Position', [10 10 600 150])
subplot(1,2,1)
box off
histogram(L3e_rates,30,'EdgeColor','none','FaceColor',[1 0 0],'FaceAlpha',1);
xlabel('ave. firing rate (Hz)')
ylabel('neuron count')
title('L3 exc.')
% grid on
xlim([0,18]) 
box off
set(gcf, 'Renderer', 'Painters');
 
 
subplot(1,2,2)
box off
histogram(L4e_rates,30,'EdgeColor','none','FaceColor',[0 0 1],'FaceAlpha',1);
xlabel('ave. firing rate (Hz)')
ylabel('neuron count')
title('L4 exc.')
% grid on
xlim([0,46]) 
box off
set(gcf, 'Renderer', 'Painters');
%% plot weight distributions
 
layer_0_layer_1_exc_weights_0s_w_normalised = (layer_0_layer_1_exc_weights_0s_w-min(layer_0_layer_1_exc_weights_0s_w))/(max(layer_0_layer_1_exc_weights_0s_w)-min(layer_0_layer_1_exc_weights_0s_w)); 
subplot(2,2,1)
hold on
histogram(layer_0_layer_1_exc_weights_0s_w_normalised,10,'EdgeColor','none','FaceColor',[0.301 0.745 0.933],'FaceAlpha',0.9);
histogram(layer_0_layer_1_exc_weights_8s_w,10,'EdgeColor','none','FaceColor',[0 0 1],'FaceAlpha',0.9);
legend('before','after')
xlabel('normalised weights')
ylabel('synapse count')
% title('layer 0 to layer 1 exc.')
% grid on
xlim([0,1]) 
box off
set(gcf, 'Renderer', 'Painters');
set(gca, 'YScale', 'log');
 
layer_1_exc_layer_2_exc_weights_0s_normalised = (layer_1_exc_layer_2_exc_weights_0s(3,:)-min(layer_1_exc_layer_2_exc_weights_0s(3,:)))/(max(layer_1_exc_layer_2_exc_weights_0s(3,:))-min(layer_1_exc_layer_2_exc_weights_0s(3,:))); 
layer_1_exc_layer_2_exc_weights_8s_normalised = (layer_1_exc_layer_2_exc_weights_8s(3,:)-min(layer_1_exc_layer_2_exc_weights_8s(3,:)))/(max(layer_1_exc_layer_2_exc_weights_8s(3,:))-min(layer_1_exc_layer_2_exc_weights_8s(3,:))); 
subplot(2,2,2)
hold on
histogram(layer_1_exc_layer_2_exc_weights_0s_normalised,20,'EdgeColor','none','FaceColor',[0.301 0.745 0.933],'FaceAlpha',1);
histogram(layer_1_exc_layer_2_exc_weights_8s_normalised,20,'EdgeColor','none','FaceColor',[0 0 1],'FaceAlpha',1);
legend('before','after')
xlabel('normalised weights')
ylabel('synapse count')
% title('layer 1 exc. to layer 2 exc.')
% grid on
xlim([0,1]) 
box off
set(gca, 'YScale', 'log');
set(gcf, 'Renderer', 'Painters');
 
layer_2_exc_layer_3_exc_weights_0s_normalised = (layer_2_exc_layer_3_exc_weights_0s(3,:)-min(layer_2_exc_layer_3_exc_weights_0s(3,:)))/(max(layer_2_exc_layer_3_exc_weights_0s(3,:))-min(layer_2_exc_layer_3_exc_weights_0s(3,:))); 
layer_2_exc_layer_3_exc_weights_8s_normalised = (layer_2_exc_layer_3_exc_weights_8s(3,:)-min(layer_2_exc_layer_3_exc_weights_8s(3,:)))/(max(layer_2_exc_layer_3_exc_weights_8s(3,:))-min(layer_2_exc_layer_3_exc_weights_8s(3,:))); 
subplot(2,2,3)
hold on
histogram(layer_2_exc_layer_3_exc_weights_0s_normalised,20,'EdgeColor','none','FaceColor',[0.301 0.745 0.933],'FaceAlpha',1);
histogram(layer_2_exc_layer_3_exc_weights_8s_normalised,20,'EdgeColor','none','FaceColor',[0 0 1],'FaceAlpha',1);
legend('before','after')
xlabel('normalised weights')
ylabel('synapse count')
% title('layer 2 exc. to layer 3 exc.')
% grid on
xlim([0,1]) 
box off
set(gca, 'YScale', 'log');
set(gcf, 'Renderer', 'Painters');
 
layer_3_exc_layer_4_exc_weights_0s_normalised = (layer_3_exc_layer_4_exc_weights_0s(3,:)-min(layer_3_exc_layer_4_exc_weights_0s(3,:)))/(max(layer_3_exc_layer_4_exc_weights_0s(3,:))-min(layer_3_exc_layer_4_exc_weights_0s(3,:))); 
layer_3_exc_layer_4_exc_weights_8s_normalised = (layer_3_exc_layer_4_exc_weights_8s(3,:)-min(layer_3_exc_layer_4_exc_weights_8s(3,:)))/(max(layer_3_exc_layer_4_exc_weights_8s(3,:))-min(layer_3_exc_layer_4_exc_weights_8s(3,:))); 
subplot(2,2,4)
hold on
histogram(layer_3_exc_layer_4_exc_weights_0s_normalised,20,'EdgeColor','none','FaceColor',[0.301 0.745 0.933],'FaceAlpha',1);
histogram(layer_3_exc_layer_4_exc_weights_8s_normalised,20,'EdgeColor','none','FaceColor',[0 0 1],'FaceAlpha',1);
legend('before','after')
xlabel('normalised weights')
ylabel('synapse count')
% title('layer 3 exc. to layer 4 exc.')
% grid on
xlim([0,1]) 
box off
set(gca, 'YScale', 'log');
set(gcf, 'Renderer', 'Painters');
%% find selective L3 neurons
 
% get average firing rates for specific image sets (see lab notebook)
L3e_rates_test_set1 = (L3e_rates_test_im5+L3e_rates_test_im7+L3e_rates_test_im9+L3e_rates_test_im14)/4;
L3e_rates_test_set2 = (L3e_rates_test_im5+L3e_rates_test_im7+L3e_rates_test_im9+L3e_rates_test_im12)/4;
L3e_rates_test_set3 = (L3e_rates_test_im5+L3e_rates_test_im7+L3e_rates_test_im6+L3e_rates_test_im12)/4;
L3e_rates_test_set4 = (L3e_rates_test_im5+L3e_rates_test_im8+L3e_rates_test_im6+L3e_rates_test_im12)/4;
 
selective = [];
 
neuron_num = [1:4096];
 
for num = neuron_num
    if L3e_rates_test_set1(num) >  L3e_rates_test_set2(num) && L3e_rates_test_set2(num) > L3e_rates_test_set3(num)  && L3e_rates_test_set3(num) > L3e_rates_test_set4(num) && L3e_rates_test_set4(num) ~= 0
        selective = [selective,num];
    end
end
%  
% for num = selective
%     figure('Position',[3000,600,200,200])
%     hold on
%     data = [L3e_rates_test_set1(num),L3e_rates_test_set2(num),L3e_rates_test_set3(num),L3e_rates_test_set4(num)];%,L3e_rates_test_set4(num)]; 
%     b = bar([1:4],data,0.9);
%     b(1).EdgeColor = 'none';
%     b(1).FaceColor = [1 0 0]
% %     b(2).EdgeColor = 'none';
% %     b(2).FaceColor = 'b';
% %     title(['exc. LIF neuron #',num2str(num),' in layer 3'])
%     ylabel('average firing rate (Hz)')
%     xlabel('image subset')
% %     legend('testing','pre-training')
%     set(gca,'xtick',1:4);
%     set(gca,'xticklabel',{'A', 'B', 'C', 'D'},'fontsize',8)
%     xlim([0.4,4.6])
%     set(gca,'OuterPosition',[0 0.01 1 1])
% %     ax = gca;
% %     ax.XGrid = 'off';
% %     ax.YGrid = 'on';
% %     scatter([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],[L3e_rates_train_im8(num),L3e_rates_train_im6(num),L3e_rates_train_im1(num),L3e_rates_train_im3(num),L3e_rates_train_im8(num),L3e_rates_train_im6(num),L3e_rates_train_im1(num),L3e_rates_train_im2(num),L3e_rates_train_im8(num),L3e_rates_train_im6(num),L3e_rates_train_im2(num),L3e_rates_train_im7(num),L3e_rates_train_im8(num),L3e_rates_train_im2(num),L3e_rates_train_im7(num),L3e_rates_train_im4(num)],'filled','b')
% end
%% trace selective L3 neurons to inputs
 
for l3_neuron = 727
    inputs_to_l3_neuron = trace_to_inputs_l3(l3_neuron,layer_2_exc_layer_3_exc_weights_8s);
    inputs_to_l2_neurons = [];
    inputs_to_l1_neurons = [];
    count = 0;
    for i=inputs_to_l3_neuron(2,:)
        count = count + 1;
        input = trace_to_inputs_l2(i,inputs_to_l3_neuron(4,count),layer_1_exc_layer_2_exc_weights_8s);
        inputs_to_l2_neurons = [inputs_to_l2_neurons,input];
    end
    count = 0;
    for i=inputs_to_l2_neurons(2,:)
        count = count + 1;
        input = trace_to_inputs_l1(i,inputs_to_l2_neurons(4,count),layer_0_layer_1_exc_weights_8s);
        inputs_to_l1_neurons = [inputs_to_l1_neurons,input];
    end
    inputs_to_l1_neurons_px = bsxfun(@rdivide,inputs_to_l1_neurons(3:4,:),12.5e-6);
    inputs_to_l1_neurons(3:4,:) = inputs_to_l1_neurons_px;
%     writematrix(inputs_to_l1_neurons,sprintf('simulation_20/weighted_inputs_l3_%s.csv',num2str(l3_neuron))) 
 
    figure
    scatter(inputs_to_l1_neurons(4,:),inputs_to_l1_neurons(3,:))
    xlim([0,256])
    ylim([0,256])
    set(gca, 'YDir','reverse')
    title(['exc. LIF neuron #',num2str(l3_neuron),' in layer 3'])
end
%% find L4 neurons to which a selective L3 neuron is connected and find PNGs
 
% find L4 neurons to which specified L3 neuron is connected
l3_neuron = 727;
syn_idx = find(layer_3_exc_layer_4_exc_weights_8s(1,:) == l3_neuron);
l4_neurons = layer_3_exc_layer_4_exc_weights_8s(2,syn_idx);
 
% turn spike trains for relevant neurons into binary series
fs = 1000;   % sampling frequency in Hz - max rate a neuron can fire is 10ms so does not need to be higher than 100Hz
T = 16;     % period of signal in s - only interested in testing part, not training 
l3_neuron_spikes = create_spike_series(l3_neuron,L3_exc_spikes_test); % get spike series for L3 neuron
% get spike series for connected L4 neurons
l4_neurons_spikes = zeros([length(l4_neurons),(fs*T)+1]); % create array to hold spike series of all L4 neurons (add one as will store neuron number in first column
count = 0;
for i = l4_neurons
    count = count + 1;
    l4_neurons_spikes(count,1) = i;
    temp = create_spike_series(i,L4_exc_spikes_test);
    l4_neurons_spikes(count,2:end) = temp(1:16000);
end
 
% % plots to visualise spikes
% figure
% plot(l3_neuron_spikes)
% xlabel('sample')
% ylabel('state')
% title(['exc. LIF neuron #',num2str(l3_neuron),' in layer 3'])    
%     
% 
% for i = [1:length(l4_neurons)]
%     figure
%     plot(l4_neurons_spikes(i,2:end))
%     xlabel('sample')
%     ylabel('state')
%     title(['exc. LIF neuron #',num2str(l4_neurons_spikes(i,1)),' in layer 4'])    
% end
% 
% % cross correlate l3 neuron spikes with each of l4 neuron spikes
% l3_l4_xcorr = zeros([length(l4_neurons),(2*(fs*T))]); % create array to hold cross correlation series of all L4 neurons with L3 neuron (store neuron number in first column
% count = 0;
% for i = l4_neurons
%     count = count+1;
%     l3_l4_xcorr(count,1) = i;
%     l3_l4_xcorr(count,2:end) = xcorr(l3_neuron_spikes,l4_neurons_spikes(count,2:end));
% end
% 
% for i = [1:length(l4_neurons)]
%     figure
%     plot(l3_l4_xcorr(i,2:end)) % specify x axis so can identify frequencies at which patterns occur
%     xlabel('sample')
%     ylabel('cross-correlation of states')
%     title(['cross-correlation of L3 exc. neuron #',num2str(l3_neuron),' and L4 exc. neuron #',num2str(l4_neurons_spikes(i,1))])    
% end    
 
% get average firing rates for specific image sets (see lab notebook)
% L4e_rates_test_set1 = (L4e_rates_test_im6+L4e_rates_test_im4+L4e_rates_test_im11+L4e_rates_test_im1)/4;
% L4e_rates_test_set2 = (L4e_rates_test_im6+L4e_rates_test_im4+L4e_rates_test_im11+L4e_rates_test_im12)/4;
% L4e_rates_test_set3 = (L4e_rates_test_im6+L4e_rates_test_im4+L4e_rates_test_im5+L4e_rates_test_im12)/4;
% L4e_rates_test_set4 = (L4e_rates_test_im6+L4e_rates_test_im14+L4e_rates_test_im5+L4e_rates_test_im12)/4;
 
% % plot firing rates of L4 neurons
% for num = l4_neurons
%     figure('Position',[3000,600,200,200])
%     hold on
%     data = [L4e_rates_test_set1(num),L4e_rates_test_set2(num),L4e_rates_test_set3(num),L4e_rates_test_set4(num)];%,L3e_rates_test_set4(num)]; 
%     b = bar([1:4],data,0.9);
%     b(1).EdgeColor = 'none';
%     b(1).FaceColor = [0 0 1]
% %     b(2).EdgeColor = 'none';
% %     b(2).FaceColor = 'b';
% %     title(['exc. LIF neuron #',num2str(num),' in layer 4'])
%     ylabel('average firing rate (Hz)')
%     xlabel('image subset')
% %     legend('testing','pre-training')
%     set(gca,'xtick',1:4);
%     set(gca,'xticklabel',{'A', 'B', 'C', 'D'},'fontsize',8)
%     xlim([0.4,4.6])
%     set(gca,'OuterPosition',[0 0.01 1 1])
% %     ax = gca;
% %     ax.XGrid = 'off';
% %     ax.YGrid = 'on';
% %     scatter([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],[L3e_rates_train_im8(num),L3e_rates_train_im6(num),L3e_rates_train_im1(num),L3e_rates_train_im3(num),L3e_rates_train_im8(num),L3e_rates_train_im6(num),L3e_rates_train_im1(num),L3e_rates_train_im2(num),L3e_rates_train_im8(num),L3e_rates_train_im6(num),L3e_rates_train_im2(num),L3e_rates_train_im7(num),L3e_rates_train_im8(num),L3e_rates_train_im2(num),L3e_rates_train_im7(num),L3e_rates_train_im4(num)],'filled','b')
% end
%% raster plots to identify PNGs
l3_spike_idx = find(l3_neuron_spikes == 1);
l3_spike_idx = l3_spike_idx.*1000/fs; % multiply by 1000 and divide by sampling frequency to get into units of ms a
l3_spike_idx = l3_spike_idx+16000;
 
l4_spike_idx = zeros(length(l4_neurons),40);
for i = [1:length(l4_neurons)]
    l4_spike_idx(i,1) = l4_neurons(i);
    if l4_neurons_spikes(i,1) == l4_neurons(i)
        l4_spike_idx_temp = (find(l4_neurons_spikes(i,2:end) == 1)-1);
        l4_spike_idx(i,2:length(l4_spike_idx_temp)+1) = l4_spike_idx_temp;
    end
end
l4_spike_idx(:,2:end) = l4_spike_idx(:,2:end).*1000/fs; % multiply by 1000 and divide by sampling frequency to get into units of ms 
l4_spike_idx(:,2:end) = l4_spike_idx(:,2:end)+16000;
labels = strings(1,length(l4_neurons)+1);
 
figure
hold on
scatter(l3_spike_idx,ones(1,length(l3_spike_idx)))
for i = [1:length(l4_neurons)]
    scatter(l4_spike_idx(i,2:end),ones(1,length(l4_spike_idx(i,2:end))).*(i+1),'r')
    labels(i+1) = num2str(l4_neurons(i));
end
ylim([0.5,26])
ylabel('neuron number')
xlabel('time (ms)')
yticks([1:length(l4_neurons)])
yticklabels([l3_neuron,l4_neurons])
ytickangle(45)
legend('L3 exc.','L4 exc.')
% ax = gca;
% ax.XGrid = 'on';
% ax.YGrid = 'off';
% set(gca,'xtick',[0:3000])
%% raster plot for PNG figure
 
figure('Renderer', 'painters', 'Position', [10 10 600 200])
hold on
scatter(l3_spike_idx,ones(1,length(l3_spike_idx)),'r')
l4_neurons_png = [734,669];
idx = [];
for i = [1:length(l4_neurons_png)]
    idx(i) = find(l4_neurons == l4_neurons_png(i));
end
count = 1;
for i = idx
    count = count + 1
    scatter(l4_spike_idx(i,2:end),ones(1,length(l4_spike_idx(i,2:end))).*count,'b')
    labels(i+1) = num2str(l4_neurons(i));
end
ylim([0,4])
ylabel('neuron number')
xlabel('time (ms)')
yticks([1:length(l4_neurons_png)+1])
yticklabels([l3_neuron,l4_neurons(idx)])
legend('layer 3','layer 4')
%% check if layer 4 neurons connected
inputs_to_l4s = [];
 for i = l4_neurons
     inputs_to_l4_idx = find(layer_4_exc_layer_4_exc_weights_8s(2,:) == i);
     inputs_to_l4 = layer_4_exc_layer_4_exc_weights_8s(1,inputs_to_l4_idx);
     for j = l4_neurons
         if inputs_to_l4 == j
             disp(i)
             disp(' is connected to ')
             disp(j)
         end
     end
 end
 %% do checking automatically
 
 % find L4 neurons to which specified L3 neuron is connected
 
for k = selective
    l3_neuron = k;
    syn_idx = find(layer_3_exc_layer_4_exc_weights_8s(1,:) == l3_neuron);
    l4_neurons = layer_3_exc_layer_4_exc_weights_8s(2,syn_idx);

    % turn spike trains for relevant neurons into binary series
    fs = 1000;   % sampling frequency in Hz - max rate a neuron can fire is 10ms so does not need to be higher than 100Hz
    T = 16;     % period of signal in s - only interested in testing part, not training 
    l3_neuron_spikes = create_spike_series(l3_neuron,L3_exc_spikes_test); % get spike series for L3 neuron
    % get spike series for connected L4 neurons
    l4_neurons_spikes = zeros([length(l4_neurons),(fs*T)+1]); % create array to hold spike series of all L4 neurons (add one as will store neuron number in first column
    count = 0;
    for i = l4_neurons
        count = count + 1;
        l4_neurons_spikes(count,1) = i;
        temp = create_spike_series(i,L4_exc_spikes_test);
        l4_neurons_spikes(count,2:end) = temp(1:16000);
    end

    inputs_to_l4s = [];
     for i = l4_neurons
         inputs_to_l4_idx = find(layer_4_exc_layer_4_exc_weights_8s(2,:) == i);
         inputs_to_l4 = layer_4_exc_layer_4_exc_weights_8s(1,inputs_to_l4_idx);
         for j = l4_neurons
             if inputs_to_l4 == j & i ~= j
                 str = ['For L3 #',num2str(k),' L4 #',num2str(i),' is connected to L4 #',num2str(j)];
                 disp(str);
             end
         end
     end
end
%% trace L4 neurons to inputs to see features they represent
 
for l4_neuron = 2037
    inputs_to_l4_neuron = trace_to_inputs_l4(l4_neuron,layer_3_exc_layer_4_exc_weights_8s);
    inputs_to_l3_neurons = [];
    inputs_to_l2_neurons = [];
    inputs_to_l1_neurons = [];
    count = 0;
    for i=inputs_to_l4_neuron(2,:)
        count = count + 1;
        input = trace_to_inputs_l3_for_l4(i,inputs_to_l4_neuron(4,count),layer_2_exc_layer_3_exc_weights_8s);
        inputs_to_l3_neurons = [inputs_to_l3_neurons,input];
    end
    count = 0;
    for i=inputs_to_l3_neurons(2,:)
        count = count + 1;
        input = trace_to_inputs_l2(i,inputs_to_l3_neurons(4,count),layer_1_exc_layer_2_exc_weights_8s);
        inputs_to_l2_neurons = [inputs_to_l2_neurons,input];
    end
    count = 0;
    for i=inputs_to_l2_neurons(2,:)
        count = count + 1;
        input = trace_to_inputs_l1(i,inputs_to_l2_neurons(4,count),layer_0_layer_1_exc_weights_8s);
        inputs_to_l1_neurons = [inputs_to_l1_neurons,input];
    end
    inputs_to_l1_neurons_px = bsxfun(@rdivide,inputs_to_l1_neurons(3:4,:),12.5e-6);
    inputs_to_l1_neurons(3:4,:) = inputs_to_l1_neurons_px;
%    writematrix(inputs_to_l1_neurons,sprintf('simulation_20/weighted_inputs_l4_simulation_%s.csv',num2str(l4_neuron)));
    figure
    scatter(inputs_to_l1_neurons(4,1:100),inputs_to_l1_neurons(3,1:100))
    xlim([0,256])
    ylim([0,256])
    set(gca, 'YDir','reverse')
    title(['exc. LIF neuron #',num2str(l4_neuron),' in layer 3'])
end 
%%
writematrix(inputs_to_l1_neurons,sprintf('simulation_20/weighted_inputs_l4_%s.csv',num2str(l4_neuron)));

%%
figure
scatter(inputs_to_l1_neurons(4,:),inputs_to_l1_neurons(3,:))
xlim([0,256])
ylim([0,256])
set(gca, 'YDir','reverse')
title(['exc. LIF neuron #',num2str(l4_neuron),' in layer 3'])
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
 
function inputs = trace_to_inputs_l4(neuron_num,layer_3_exc_layer_4_exc_weights_8_0s)
    % find synapses from previous layer neurons to this neuron
    syn_idx = find(layer_3_exc_layer_4_exc_weights_8_0s(2,:) == neuron_num);
    % store index, pre and post indices and weight
    inputs = [syn_idx;layer_3_exc_layer_4_exc_weights_8_0s(:,syn_idx)];% layer_2_exc_layer_3_exc_weights_8_0s(1,syn_idx);layer_2_exc_layer_3_exc_weights_8_0s(2,syn_idx);layer_2_exc_layer_3_exc_weights_8_0s(3,syn_idx)];
    % sort by weights in descending order 
    inputs = transpose(inputs);
    inputs = sortrows(inputs,4,'descend');
    inputs = transpose(inputs);
end
 
function inputs = trace_to_inputs_l3_for_l4(neuron_num,weight,layer_2_exc_layer_3_exc_weights_8_0s)
    % find synapses from previous layer neurons to this neuron
    syn_idx = find(layer_2_exc_layer_3_exc_weights_8_0s(2,:) == neuron_num);
    % store index, pre and post indices and weight
    inputs = [syn_idx;layer_2_exc_layer_3_exc_weights_8_0s(:,syn_idx);layer_2_exc_layer_3_exc_weights_8_0s(1,syn_idx);layer_2_exc_layer_3_exc_weights_8_0s(2,syn_idx);layer_2_exc_layer_3_exc_weights_8_0s(3,syn_idx)+weight];
    % sort by weights in descending order 
    inputs = transpose(inputs);
    inputs = sortrows(inputs,4,'descend');
    inputs = transpose(inputs);
end
% fun
 
function inputs = trace_to_inputs_l3(neuron_num,layer_2_exc_layer_3_exc_weights_8_0s)
    % find synapses from previous layer neurons to this neuron
    syn_idx = find(layer_2_exc_layer_3_exc_weights_8_0s(2,:) == neuron_num);
    % store index, pre and post indices and weight
    inputs = [syn_idx;layer_2_exc_layer_3_exc_weights_8_0s(:,syn_idx)];% layer_2_exc_layer_3_exc_weights_8_0s(1,syn_idx);layer_2_exc_layer_3_exc_weights_8_0s(2,syn_idx);layer_2_exc_layer_3_exc_weights_8_0s(3,syn_idx)];
    % sort by weights in descending order 
    inputs = transpose(inputs);
    inputs = sortrows(inputs,4,'descend');
    inputs = transpose(inputs);
end
% function inputs = trace_to_inputs_l3(neuron_num,layer_2_exc_layer_3_exc_weights_8_0s)
%     % get location of neuron
%     x_loc = mod(neuron_num,64)*5e-5;
%     y_loc = floor(neuron_num/64)*5e-5;
%     % find synapses from previous layer neurons to this neuron
%     syn_idx = find(round(layer_2_exc_layer_3_exc_weights_8_0s(3,:),5) == round(x_loc,5) & round(layer_2_exc_layer_3_exc_weights_8_0s(4,:),5) == round(y_loc,5));
%     % store index, weight and x and y locations of all connected neurons
%     inputs = [syn_idx;layer_2_exc_layer_3_exc_weights_8_0s(5,syn_idx);layer_2_exc_layer_3_exc_weights_8_0s(1,syn_idx);layer_2_exc_layer_3_exc_weights_8_0s(2,syn_idx)];
%     % sort by weights in descending order 
%     inputs = transpose(inputs);
%     inputs = sortrows(inputs,2,'descend');
%     inputs = transpose(inputs);
% end
 
 
 
% function to trace synapses from a neuron to neurons it's connected to in
% the previous layer
function inputs = trace_to_inputs_l2(neuron_num,weight,layer_1_exc_layer_2_exc_weights_8_0s)
    % find synapses from previous layer neurons to this neuron
    syn_idx = find(layer_1_exc_layer_2_exc_weights_8_0s(2,:) == neuron_num);
    % store index, weight and x and y locations of all connected neurons
    inputs = [syn_idx; layer_1_exc_layer_2_exc_weights_8_0s(1,syn_idx);layer_1_exc_layer_2_exc_weights_8_0s(2,syn_idx);layer_1_exc_layer_2_exc_weights_8_0s(3,syn_idx)+weight];
    % sort by weights in descending order 
    inputs = transpose(inputs);
    inputs = sortrows(inputs,4,'descend');
    inputs = transpose(inputs);
end
% function inputs = trace_to_inputs_l2(x_loc,y_loc,weight,layer_1_exc_layer_2_exc_weights_8_0s)
%     % find synapses from previous layer neurons to this neuron
%     syn_idx = find(round(layer_1_exc_layer_2_exc_weights_8_0s(3,:),5) == round(x_loc,5) & round(layer_1_exc_layer_2_exc_weights_8_0s(4,:),5) == round(y_loc,5));
%     % store index, weight and x and y locations of all connected neurons
%     inputs = [syn_idx;layer_1_exc_layer_2_exc_weights_8_0s(5,syn_idx).*weight;layer_1_exc_layer_2_exc_weights_8_0s(1,syn_idx);layer_1_exc_layer_2_exc_weights_8_0s(2,syn_idx)];
%     % sort by weights in descending order 
%     inputs = transpose(inputs);
%     inputs = sortrows(inputs,2,'descend');
%     inputs = transpose(inputs);
% end
 
 
 
% function to trace synapses from a neuron to neurons it's connected to in
% the previous layer
function inputs = trace_to_inputs_l1(neuron_num,weight,layer_0_layer_1_exc_weights_8_0s)
    % find synapses from previous layer neurons to this neuron
    syn_idx = find(layer_0_layer_1_exc_weights_8_0s(2,:) == neuron_num);
    % store index, weight and x and y locations and filter number of all connected neurons
   inputs = [syn_idx;layer_0_layer_1_exc_weights_8_0s(6,syn_idx)+weight;layer_0_layer_1_exc_weights_8_0s(3,syn_idx);layer_0_layer_1_exc_weights_8_0s(4,syn_idx);layer_0_layer_1_exc_weights_8_0s(5,syn_idx)];
end
% function inputs = trace_to_inputs_poisson(x_loc,y_loc,weight,layer_0_layer_1_exc_weights_8_0s)
%     % find synapses from previous layer neurons to this neuron
%     syn_idx = find(round(layer_0_layer_1_exc_weights_8_0s(3,:),5) == round(x_loc,5) & round(layer_0_layer_1_exc_weights_8_0s(4,:),5) == round(y_loc,5));
%     % store index, weight and x and y locations and filter number of all connected neurons
%     inputs = [syn_idx;layer_0_layer_1_exc_weights_8_0s(6,syn_idx).*weight;layer_0_layer_1_exc_weights_8_0s(1,syn_idx);layer_0_layer_1_exc_weights_8_0s(2,syn_idx);layer_0_layer_1_exc_weights_8_0s(5,syn_idx)];
% end
 
 
 
 
function l0_inputs = trace_L3_to_L0_weighted(l3_neuron,layer_2_exc_layer_3_exc_weights_8_0s,layer_1_exc_layer_2_exc_weights_8_0s,layer_0_layer_1_exc_weights_8_0s)
    l2_inputs = trace_to_inputs_l3(l3_neuron,layer_2_exc_layer_3_exc_weights_8_0s);
    l1_inputs = [];
    l0_inputs = [];
    for i=length(l2_inputs(1,:))
        idx_l2 = l2_inputs(3,i);
        l1_input = trace_to_inputs_l2(idx_l2,l2_inputs(2,i),layer_1_exc_layer_2_exc_weights_8_0s);
        l1_inputs = [l1_inputs,l1_input];
    end
    for i=length(l1_inputs(1,:))
        idx_l1 = l1_inputs(3,i);
        l0_input = trace_to_inputs_l1(idx_l1,l1_inputs(2,i),layer_0_layer_1_exc_weights_8_0s);
        l0_inputs = [l0_inputs,l0_input];
    end
end
% function l0_inputs = trace_L3_to_L0_weighted(l3_neuron,layer_2_exc_layer_3_exc_weights_8_0s,layer_1_exc_layer_2_exc_weights_8_0s,layer_0_layer_1_exc_weights_8_0s)
%     l2_inputs = trace_to_inputs_l3(l3_neuron,layer_2_exc_layer_3_exc_weights_8_0s);
%     l1_inputs = [];
%     l0_inputs = [];
%     for i=length(l2_inputs(1,:))
%         x_l2 = l2_inputs(3,i);
%         y_l2 = l2_inputs(4,i);
%         l1_input = trace_to_inputs_l2(x_l2,y_l2,l2_inputs(2,i),layer_1_exc_layer_2_exc_weights_8_0s);
%         l1_inputs = [l1_inputs,l1_input];
%     end
%     for i=length(l1_inputs(1,:))
%         x_l1 = l1_inputs(3,i);
%         y_l1 = l1_inputs(4,i);
%         l0_input = trace_to_inputs_poisson(x_l1,y_l1,l1_inputs(2,i),layer_0_layer_1_exc_weights_8_0s);
%         l0_inputs = [l0_inputs,l0_input];
%     end
% end
 
function spike_series = create_spike_series(neuron,spike_layer_variable)
    fs = 1000;   % sampling frequency in Hz - max rate a neuron can fire is 10ms so does not need to be higher than 100Hz
    T = 16;     % period of signal in s - only interested in testing part, not training 
    spike_series = zeros([1,fs*T]);
    for i = [1:length(spike_layer_variable)]
        if spike_layer_variable(1,i) == neuron
            idx = cast((spike_layer_variable(2,i)-16)*fs,'int64');
            spike_series(idx+1) = 1;
        end
    end
end
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


