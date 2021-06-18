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
%% create matrix to track selective neurons
selectivity_stats = zeros(2,8);
%% concave top
id = 1;
L3e_rates_test_set1 = (L3e_rates_test_im12+L3e_rates_test_im1+L3e_rates_test_im15+L3e_rates_test_im8)/4;
L3e_rates_test_set2 = (L3e_rates_test_im12+L3e_rates_test_im1+L3e_rates_test_im15+L3e_rates_test_im7)/4;
L3e_rates_test_set3 = (L3e_rates_test_im12+L3e_rates_test_im1+L3e_rates_test_im13+L3e_rates_test_im7)/4;
L3e_rates_test_set4 = (L3e_rates_test_im12+L3e_rates_test_im5+L3e_rates_test_im13+L3e_rates_test_im7)/4;
x_lower = 75;
x_upper = 175;
y_lower = 0;
y_upper = 100;



% apparently selective neurons
selective = [];
neuron_num = [1:4096];
for num = neuron_num
    if L3e_rates_test_set1(num) >  L3e_rates_test_set2(num) && L3e_rates_test_set2(num) > L3e_rates_test_set3(num)  && L3e_rates_test_set3(num) > L3e_rates_test_set4(num) && L3e_rates_test_set4(num) ~= 0
        selective = [selective,num];
    end
end
selectivity_stats(1,id) = length(selective);

% confirm selective neurons

confirmed_count = 0;
for l3_neuron = selective
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
    mean_x = mean(inputs_to_l1_neurons(3,:));
    mean_y = mean(inputs_to_l1_neurons(4,:));
    if mean_x >= x_lower && mean_x <= x_upper && mean_y >= y_lower && mean_y <= y_upper
        confirmed_count = confirmed_count +1;
    end
end
selectivity_stats(2,id) = confirmed_count;
%% concave bottom
id = 2;
L3e_rates_test_set1 = (L3e_rates_test_im1+L3e_rates_test_im2+L3e_rates_test_im4+L3e_rates_test_im11)/4;
L3e_rates_test_set2 = (L3e_rates_test_im1+L3e_rates_test_im2+L3e_rates_test_im4+L3e_rates_test_im13)/4;
L3e_rates_test_set3 = (L3e_rates_test_im1+L3e_rates_test_im2+L3e_rates_test_im5+L3e_rates_test_im13)/4;
L3e_rates_test_set4 = (L3e_rates_test_im1+L3e_rates_test_im3+L3e_rates_test_im5+L3e_rates_test_im13)/4;
x_lower = 75;
x_upper = 175;
y_lower = 100;
y_upper = 200;



% apparently selective neurons
selective = [];
neuron_num = [1:4096];
for num = neuron_num
    if L3e_rates_test_set1(num) >  L3e_rates_test_set2(num) && L3e_rates_test_set2(num) > L3e_rates_test_set3(num)  && L3e_rates_test_set3(num) > L3e_rates_test_set4(num) && L3e_rates_test_set4(num) ~= 0
        selective = [selective,num];
    end
end
selectivity_stats(1,id) = length(selective);

% confirm selective neurons

confirmed_count = 0;
for l3_neuron = selective
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
    mean_x = mean(inputs_to_l1_neurons(3,:));
    mean_y = mean(inputs_to_l1_neurons(4,:));
    if mean_x >= x_lower && mean_x <= x_upper && mean_y >= y_lower && mean_y <= y_upper
        confirmed_count = confirmed_count +1;
    end
end
selectivity_stats(2,id) = confirmed_count;
%% concave left

id = 3;
L3e_rates_test_set1 = (L3e_rates_test_im3+L3e_rates_test_im4+L3e_rates_test_im13+L3e_rates_test_im10)/4;
L3e_rates_test_set2 = (L3e_rates_test_im3+L3e_rates_test_im4+L3e_rates_test_im13+L3e_rates_test_im15)/4;
L3e_rates_test_set3 = (L3e_rates_test_im3+L3e_rates_test_im4+L3e_rates_test_im7+L3e_rates_test_im15)/4;
L3e_rates_test_set4 = (L3e_rates_test_im3+L3e_rates_test_im12+L3e_rates_test_im7+L3e_rates_test_im15)/4;
x_lower = 50;
x_upper = 150;
y_lower = 75;
y_upper = 175;



% apparently selective neurons
selective = [];
neuron_num = [1:4096];
for num = neuron_num
    if L3e_rates_test_set1(num) >  L3e_rates_test_set2(num) && L3e_rates_test_set2(num) > L3e_rates_test_set3(num)  && L3e_rates_test_set3(num) > L3e_rates_test_set4(num) && L3e_rates_test_set4(num) ~= 0
        selective = [selective,num];
    end
end
selectivity_stats(1,id) = length(selective);

% confirm selective neurons

confirmed_count = 0;
for l3_neuron = selective
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
    mean_x = mean(inputs_to_l1_neurons(3,:));
    mean_y = mean(inputs_to_l1_neurons(4,:));
    if mean_x >= x_lower && mean_x <= x_upper && mean_y >= y_lower && mean_y <= y_upper
        confirmed_count = confirmed_count +1;
    end
end
selectivity_stats(2,id) = confirmed_count;
%% concave right

id = 4;
L3e_rates_test_set1 = (L3e_rates_test_im16+L3e_rates_test_im15+L3e_rates_test_im13+L3e_rates_test_im6)/4;
L3e_rates_test_set2 = (L3e_rates_test_im16+L3e_rates_test_im15+L3e_rates_test_im13+L3e_rates_test_im12)/4;
L3e_rates_test_set3 = (L3e_rates_test_im16+L3e_rates_test_im15+L3e_rates_test_im14+L3e_rates_test_im12)/4;
L3e_rates_test_set4 = (L3e_rates_test_im16+L3e_rates_test_im7+L3e_rates_test_im14+L3e_rates_test_im12)/4;
x_lower = 125;
x_upper = 225;
y_lower = 75;
y_upper = 175;



% apparently selective neurons
selective = [];
neuron_num = [1:4096];
for num = neuron_num
    if L3e_rates_test_set1(num) >  L3e_rates_test_set2(num) && L3e_rates_test_set2(num) > L3e_rates_test_set3(num)  && L3e_rates_test_set3(num) > L3e_rates_test_set4(num) && L3e_rates_test_set4(num) ~= 0
        selective = [selective,num];
    end
end
selectivity_stats(1,id) = length(selective);

% confirm selective neurons

confirmed_count = 0;
for l3_neuron = selective
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
    mean_x = mean(inputs_to_l1_neurons(3,:));
    mean_y = mean(inputs_to_l1_neurons(4,:));
    if mean_x >= x_lower && mean_x <= x_upper && mean_y >= y_lower && mean_y <= y_upper
        confirmed_count = confirmed_count +1;
    end
end
selectivity_stats(2,id) = confirmed_count;
%% convex top

id = 5;
L3e_rates_test_set1 = (L3e_rates_test_im7+L3e_rates_test_im13+L3e_rates_test_im5+L3e_rates_test_im10)/4;
L3e_rates_test_set2 = (L3e_rates_test_im7+L3e_rates_test_im13+L3e_rates_test_im5+L3e_rates_test_im12)/4;
L3e_rates_test_set3 = (L3e_rates_test_im7+L3e_rates_test_im13+L3e_rates_test_im1+L3e_rates_test_im12)/4;
L3e_rates_test_set4 = (L3e_rates_test_im7+L3e_rates_test_im15+L3e_rates_test_im1+L3e_rates_test_im12)/4;
x_lower = 75;
x_upper = 175;
y_lower = 50;
y_upper = 150;



% apparently selective neurons
selective = [];
neuron_num = [1:4096];
for num = neuron_num
    if L3e_rates_test_set1(num) >  L3e_rates_test_set2(num) && L3e_rates_test_set2(num) > L3e_rates_test_set3(num)  && L3e_rates_test_set3(num) > L3e_rates_test_set4(num) && L3e_rates_test_set4(num) ~= 0
        selective = [selective,num];
    end
end
selectivity_stats(1,id) = length(selective);

% confirm selective neurons

confirmed_count = 0;
for l3_neuron = selective
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
    mean_x = mean(inputs_to_l1_neurons(3,:));
    mean_y = mean(inputs_to_l1_neurons(4,:));
    if mean_x >= x_lower && mean_x <= x_upper && mean_y >= y_lower && mean_y <= y_upper
        confirmed_count = confirmed_count +1;
    end
end
selectivity_stats(2,id) = confirmed_count;
%% convex bottom


id = 6;
L3e_rates_test_set1 = (L3e_rates_test_im13+L3e_rates_test_im5+L3e_rates_test_im3+L3e_rates_test_im12)/4;
L3e_rates_test_set2 = (L3e_rates_test_im13+L3e_rates_test_im5+L3e_rates_test_im3+L3e_rates_test_im1)/4;
L3e_rates_test_set3 = (L3e_rates_test_im13+L3e_rates_test_im5+L3e_rates_test_im2+L3e_rates_test_im1)/4;
L3e_rates_test_set4 = (L3e_rates_test_im13+L3e_rates_test_im4+L3e_rates_test_im2+L3e_rates_test_im1)/4;
x_lower = 75;
x_upper = 175;
y_lower = 150;
y_upper = 250;



% apparently selective neurons
selective = [];
neuron_num = [1:4096];
for num = neuron_num
    if L3e_rates_test_set1(num) >  L3e_rates_test_set2(num) && L3e_rates_test_set2(num) > L3e_rates_test_set3(num)  && L3e_rates_test_set3(num) > L3e_rates_test_set4(num) && L3e_rates_test_set4(num) ~= 0
        selective = [selective,num];
    end
end
selectivity_stats(1,id) = length(selective);

% confirm selective neurons

confirmed_count = 0;
for l3_neuron = selective
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
    mean_x = mean(inputs_to_l1_neurons(3,:));
    mean_y = mean(inputs_to_l1_neurons(4,:));
    if mean_x >= x_lower && mean_x <= x_upper && mean_y >= y_lower && mean_y <= y_upper
        confirmed_count = confirmed_count +1;
    end
end
selectivity_stats(2,id) = confirmed_count;
%% convex left

id = 7;
L3e_rates_test_set1 = (L3e_rates_test_im15+L3e_rates_test_im7+L3e_rates_test_im12+L3e_rates_test_im16)/4;
L3e_rates_test_set2 = (L3e_rates_test_im15+L3e_rates_test_im7+L3e_rates_test_im12+L3e_rates_test_im3)/4;
L3e_rates_test_set3 = (L3e_rates_test_im15+L3e_rates_test_im7+L3e_rates_test_im4+L3e_rates_test_im3)/4;
L3e_rates_test_set4 = (L3e_rates_test_im15+L3e_rates_test_im13+L3e_rates_test_im4+L3e_rates_test_im3)/4;
x_lower = 0;
x_upper = 150;
y_lower = 75;
y_upper = 175;



% apparently selective neurons
selective = [];
neuron_num = [1:4096];
for num = neuron_num
    if L3e_rates_test_set1(num) >  L3e_rates_test_set2(num) && L3e_rates_test_set2(num) > L3e_rates_test_set3(num)  && L3e_rates_test_set3(num) > L3e_rates_test_set4(num) && L3e_rates_test_set4(num) ~= 0
        selective = [selective,num];
    end
end
selectivity_stats(1,id) = length(selective);

% confirm selective neurons

confirmed_count = 0;
for l3_neuron = selective
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
    mean_x = mean(inputs_to_l1_neurons(3,:));
    mean_y = mean(inputs_to_l1_neurons(4,:));
    if mean_x >= x_lower && mean_x <= x_upper && mean_y >= y_lower && mean_y <= y_upper
        confirmed_count = confirmed_count +1;
    end
end
selectivity_stats(2,id) = confirmed_count;
%% convex right


id = 8;
L3e_rates_test_set1 = (L3e_rates_test_im12+L3e_rates_test_im14+L3e_rates_test_im7+L3e_rates_test_im5)/4;
L3e_rates_test_set2 = (L3e_rates_test_im12+L3e_rates_test_im14+L3e_rates_test_im7+L3e_rates_test_im16)/4;
L3e_rates_test_set3 = (L3e_rates_test_im12+L3e_rates_test_im14+L3e_rates_test_im15+L3e_rates_test_im16)/4;
L3e_rates_test_set4 = (L3e_rates_test_im12+L3e_rates_test_im13+L3e_rates_test_im15+L3e_rates_test_im16)/4;
x_lower = 150;
x_upper = 250;
y_lower = 75;
y_upper = 175;



% apparently selective neurons
selective = [];
neuron_num = [1:4096];
for num = neuron_num
    if L3e_rates_test_set1(num) >  L3e_rates_test_set2(num) && L3e_rates_test_set2(num) > L3e_rates_test_set3(num)  && L3e_rates_test_set3(num) > L3e_rates_test_set4(num) && L3e_rates_test_set4(num) ~= 0
        selective = [selective,num];
    end
end
selectivity_stats(1,id) = length(selective);

% confirm selective neurons

confirmed_count = 0;
for l3_neuron = selective
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
    mean_x = mean(inputs_to_l1_neurons(3,:));
    mean_y = mean(inputs_to_l1_neurons(4,:));
    if mean_x >= x_lower && mean_x <= x_upper && mean_y >= y_lower && mean_y <= y_upper
        confirmed_count = confirmed_count +1;
    end
end
selectivity_stats(2,id) = confirmed_count;
%% plot histogram

figure('Position',[3000,600,200,200])
hold on
grid off
data = selectivity_stats;
b = bar(data(1,:));
% b(1).EdgeColor = 'none';
% b(1).FaceColor = [1 0 0];
% b(2).EdgeColor = 'none';
% b(2).FaceColor = 'b';
ylabel('feature')
xlabel('neuron count')
legend('apparent selectivity','confirmed selectivity')
set(gca,'xtick',1:16);
set(gca,'xticklabel',{'concave top', 'concave bottom', 'concave left', 'concave right','convex top', 'convex bottom', 'convex left', 'convex right'},'fontsize',8)
% xlim([0.4,8.6])
set(gca,'OuterPosition',[0 0.01 1 1])
ax = gca;

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
 
 
 
 