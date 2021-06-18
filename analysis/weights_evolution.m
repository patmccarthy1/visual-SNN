clear all; close all; clc
simulation_num = '4';
%% read spike data
% spike data
L0_spikes_train = readmatrix(sprintf('../output_data/simulation_%s/layer_0_train_spikes.csv',simulation_num));
% L0_spikes_test = readmatrix(sprintf('../output_data/simulation_%s/layer_0_test_spikes.csv',simulation_num));
L1_exc_spikes_train = readmatrix(sprintf('../output_data/simulation_%s/layer_1_excitatory_train_spikes.csv',simulation_num));
% L1_exc_spikes_test = readmatrix(sprintf('../output_data/simulation_%s/layer_1_excitatory_test_spikes.csv',simulation_num));
L1_inh_spikes_train = readmatrix(sprintf('../output_data/simulation_%s/layer_1_inhibitory_train_spikes.csv',simulation_num));
% L1_inh_spikes_test = readmatrix(sprintf('../output_data/simulation_%s/layer_1_inhitatory_test_spikes.csv',simulation_num));
L2_exc_spikes_train = readmatrix(sprintf('../output_data/simulation_%s/layer_2_excitatory_train_spikes.csv',simulation_num));
% L2_exc_spikes_test = readmatrix(sprintf('../output_data/simulation_%s/layer_2_excitatory_test_spikes.csv',simulation_num));
L2_inh_spikes_train = readmatrix(sprintf('../output_data/simulation_%s/layer_2_inhibitory_train_spikes.csv',simulation_num));
% L2_inh_spikes_test = readmatrix(sprintf('../output_data/simulation_%s/layer_2_inhitatory_test_spikes.csv',simulation_num));
L3_exc_spikes_train = readmatrix(sprintf('../output_data/simulation_%s/layer_3_excitatory_train_spikes.csv',simulation_num));
% L3_exc_spikes_test = readmatrix(sprintf('../output_data/simulation_%s/layer_3_excitatory_test_spikes.csv',simulation_num));
L3_inh_spikes_train = readmatrix(sprintf('../output_data/simulation_%s/layer_3_inhibitory_train_spikes.csv',simulation_num));
% L3_inh_spikes_test = readmatrix(sprintf('../output_data/simulation_%s/layer_3_inhitatory_test_spikes.csv',simulation_num));
L4_exc_spikes_train = readmatrix(sprintf('../output_data/simulation_%s/layer_4_excitatory_train_spikes.csv',simulation_num));
% L4_exc_spikes_test = readmatrix(sprintf('../output_data/simulation_%s/layer_4_excitatory_test_spikes.csv',simulation_num));
L4_inh_spikes_train = readmatrix(sprintf('../output_data/simulation_%s/layer_4_inhibitory_train_spikes.csv',simulation_num));
% L4_inh_spikes_test = readmatrix(sprintf('../output_data/simulation_%s/layer_4_inhitatory_test_spikes.csv',simulation_num));
%% read weight data
layer_3_exc_layer_4_exc_weights_im0_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im0_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im0_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im0_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im0_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im0_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im0_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im0_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im0_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im0_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im1_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im1_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im1_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im1_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im1_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im1_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im1_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im1_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im1_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im1_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im2_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im2_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im2_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im2_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im2_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im2_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im2_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im2_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im2_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im2_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im3_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im3_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im3_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im3_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im3_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im3_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im3_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im3_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im3_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im3_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im4_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im4_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im4_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im4_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im4_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im4_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im4_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im4_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im4_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im4_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im5_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im5_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im5_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im5_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im5_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im5_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im5_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im5_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im5_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im5_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im6_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im6_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im6_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im6_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im6_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im6_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im6_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im6_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im6_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im6_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im7_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im7_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im7_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im7_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im7_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im7_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im7_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im7_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im7_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im7_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im8_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im8_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im8_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im8_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im8_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im8_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im8_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im8_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im8_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im8_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im9_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im9_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im9_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im9_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im9_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im9_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im9_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im9_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im9_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im9_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im10_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im10_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im10_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im10_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im10_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im10_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im10_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im10_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im10_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im10_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im11_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im11_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im11_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im11_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im11_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im11_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im11_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im11_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im11_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im11_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im12_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im12_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im12_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im12_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im12_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im12_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im12_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im12_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im12_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im12_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im13_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im13_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im13_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im13_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im13_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im13_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im13_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im13_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im13_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im13_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im14_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im14_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im14_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im14_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im14_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im14_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im14_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im14_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im14_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im14_1s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im15_0_2s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im15_0_2s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im15_0_4s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im15_0_4s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im15_0_6s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im15_0_6s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im15_0_8s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im15_0_8s.csv',simulation_num));
layer_3_exc_layer_4_exc_weights_im15_1s = readmatrix(sprintf('../output_data/simulation_%s/layer_3_exc_layer_4_exc_weights_im15_1s.csv',simulation_num));
%%
idx = [6001];

weights = [];
count = 0;
for i = idx
    count = count + 1;
    weights(count,:) = [
        layer_3_exc_layer_4_exc_weights_im0_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im0_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im0_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im0_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im0_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im1_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im1_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im1_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im1_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im1_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im2_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im2_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im2_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im2_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im2_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im3_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im3_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im3_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im3_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im3_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im4_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im4_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im4_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im4_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im4_1s(6,i), 
        layer_3_exc_layer_4_exc_weights_im5_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im5_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im5_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im5_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im5_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im6_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im6_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im6_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im6_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im6_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im7_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im7_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im7_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im7_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im7_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im8_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im8_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im8_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im8_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im8_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im9_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im9_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im9_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im9_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im9_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im10_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im10_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im10_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im10_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im10_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im11_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im11_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im11_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im11_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im11_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im12_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im12_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im12_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im12_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im12_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im13_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im13_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im13_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im13_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im13_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im14_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im14_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im14_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im14_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im14_1s(6,i), 
        layer_3_exc_layer_4_exc_weights_im15_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im15_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im15_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im15_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im15_1s(6,i)];
end

figure
plot(weights)
%% put weights for particular synapse into vector
close all;
idx = [750,860,4249,6782,1,2,4,499];

colours = [[0 0 1];[0.301 0.745 0.933];[0 0.3 0.7];[0 0.8 0.8];[1 0 0];[1 0 0.1];[1 0.2 0.1];[1 0.4 0.6]];
weights = [];
count = 0;
for i = idx
    count = count + 1;
    weights(count,:) = [
        layer_3_exc_layer_4_exc_weights_im0_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im0_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im0_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im0_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im0_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im1_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im1_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im1_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im1_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im1_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im2_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im2_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im2_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im2_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im2_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im3_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im3_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im3_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im3_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im3_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im4_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im4_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im4_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im4_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im4_1s(6,i), 
        layer_3_exc_layer_4_exc_weights_im5_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im5_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im5_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im5_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im5_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im6_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im6_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im6_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im6_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im6_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im7_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im7_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im7_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im7_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im7_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im8_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im8_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im8_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im8_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im8_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im9_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im9_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im9_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im9_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im9_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im10_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im10_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im10_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im10_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im10_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im11_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im11_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im11_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im11_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im11_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im12_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im12_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im12_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im12_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im12_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im13_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im13_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im13_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im13_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im13_1s(6,i),
        layer_3_exc_layer_4_exc_weights_im14_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im14_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im14_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im14_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im14_1s(6,i), 
        layer_3_exc_layer_4_exc_weights_im15_0_2s(6,i),
        layer_3_exc_layer_4_exc_weights_im15_0_4s(6,i),
        layer_3_exc_layer_4_exc_weights_im15_0_6s(6,i),
        layer_3_exc_layer_4_exc_weights_im15_0_8s(6,i),
        layer_3_exc_layer_4_exc_weights_im15_1s(6,i)];
end

change = diff(weights,1,2);
% plot weights over time

% normalized

weights = normalize(weights,2,'norm',Inf);

figure('Renderer', 'painters', 'Position', [10 10 600 600])

subplot(4,1,1)
hold on
count = 0
for i = [1:4]
    count = count + 1;
    plot([1:length(weights(1,:))],weights(i,:),'LineWidth',2,'Color',colours(i,:))
end
xticks([1:5:80])
xticklabels([0:17])
ylabel('normalised weight')
xlabel('time (s)')
set(gca,'YGrid','off','XGrid','on');

subplot(4,1,2)
hold on
xlim([1,79])
count = 0;
for i = [1:4]
    count = count + 1;
    plot([1:length(change(1,:))],change(i,:),'LineWidth',2,'Color',colours(i,:))
end
xticks([1:5:80])
xticklabels([0:17])
ylabel('diff. nnormalised weight')
xlabel('time (s)')
xlim([1,79])
set(gca,'YGrid','off','XGrid','on');
box off

subplot(4,1,3)
hold on
count = 0;
for i = [5:8]
    count = count + 1;
    plot([1:length(weights(1,:))],weights(i,:),'LineWidth',2,'Color',colours(i,:))
end
xticks([1:5:80])
xticklabels([0:17])
ylabel('normalised weight')
xlabel('time (s)')
set(gca,'YGrid','off','XGrid','on');

subplot(4,1,4)
hold on
xlim([1,79])
count = 0;
for i = [5:8]
    count = count + 1;
    plot([1:length(change(1,:))],change(i,:),'LineWidth',2,'Color',colours(i,:))
end
xticks([1:5:80])
xticklabels([0:17])
ylabel('diff. in normalised weight')
xlabel('time (s)')
xlim([1,79])
set(gca,'YGrid','off','XGrid','on');
box off

% lg = legend('before','after')
% newPosition = [0.9,0.475,0.05 0.05];
% newUnits = 'normalized';
% set(lg,'Position', newPosition,'Units', newUnits);
