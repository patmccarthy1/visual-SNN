% this file is for quick plots - for plots for final report redo in Python
% with matplotlib/ seaborn (see joint plots)
%% read in data

% spike data
L0_spikes = readmatrix('../output_data/simulation_3/layer_0_spikes.csv');
L1_exc_spikes = readmatrix('../output_data/simulation_3/layer_1_excitatory_spikes.csv');
L1_inh_spikes = readmatrix('../output_data/simulation_3/layer_1_inhibitory_spikes.csv');
L2_exc_spikes = readmatrix('../output_data/simulation_3/layer_2_excitatory_spikes.csv');
L2_inh_spikes = readmatrix('../output_data/simulation_3/layer_2_inhibitory_spikes.csv');
L3_exc_spikes = readmatrix('../output_data/simulation_3/layer_3_excitatory_spikes.csv');
L3_inh_spikes = readmatrix('../output_data/simulation_3/layer_3_inhibitory_spikes.csv');
L4_exc_spikes = readmatrix('../output_data/simulation_3/layer_4_excitatory_spikes.csv');
L4_inh_spikes = readmatrix('../output_data/simulation_3/layer_4_inhibitory_spikes.csv');

% % weight data
% L1_exc_L2_exc_weights =  csvread('../output_data/layer_1_exc_layer_2_exc_weights.csv');
% %sorted_L1_exc_L2_exc_weights = sortcols(L1_exc_L2_exc_weights,6,'descend');

%% separate spikes for each image

% find indices of spikess after each image presented
L3_exc_spikes_im1_idx = find(L3_exc_spikes(2,:)>1,1);
L3_exc_spikes_im1 = L3_exc_spikes(:,1:L3_exc_spikes_im1_idx);
L3_exc_spikes_im2_idx = find(L3_exc_spikes(2,:)>2,1);
L3_exc_spikes_im2 = L3_exc_spikes(:,L3_exc_spikes_im1_idx:L3_exc_spikes_im2_idx);
L3_exc_spikes_im3_idx = find(L3_exc_spikes(2,:)>3,1);
L3_exc_spikes_im3 = L3_exc_spikes(:,L3_exc_spikes_im2_idx:L3_exc_spikes_im3_idx);
L3_exc_spikes_im4_idx = find(L3_exc_spikes(2,:)>4,1);
L3_exc_spikes_im4 = L3_exc_spikes(:,L3_exc_spikes_im3_idx:L3_exc_spikes_im4_idx);
L3_exc_spikes_im5_idx = find(L3_exc_spikes(2,:)>5,1);
L3_exc_spikes_im5 = L3_exc_spikes(:,L3_exc_spikes_im4_idx:L3_exc_spikes_im5_idx);
L3_exc_spikes_im6_idx = find(L3_exc_spikes(2,:)>6,1);
L3_exc_spikes_im6 = L3_exc_spikes(:,L3_exc_spikes_im5_idx:L3_exc_spikes_im6_idx);
L3_exc_spikes_im7_idx = find(L3_exc_spikes(2,:)>7,1);
L3_exc_spikes_im7 = L3_exc_spikes(:,L3_exc_spikes_im6_idx:L3_exc_spikes_im7_idx);
L3_exc_spikes_im8_idx = find(L3_exc_spikes(2,:)>8,1);
L3_exc_spikes_im8 = L3_exc_spikes(:,L3_exc_spikes_im7_idx:L3_exc_spikes_im8_idx);
L3_exc_spikes_im9_idx = find(L3_exc_spikes(2,:)>9,1);
L3_exc_spikes_im9 = L3_exc_spikes(:,L3_exc_spikes_im8_idx:L3_exc_spikes_im9_idx);
L3_exc_spikes_im10_idx = find(L3_exc_spikes(2,:)>10,1);
L3_exc_spikes_im10 = L3_exc_spikes(:,L3_exc_spikes_im9_idx:L3_exc_spikes_im10_idx);
L3_exc_spikes_im11_idx = find(L3_exc_spikes(2,:)>11,1);
L3_exc_spikes_im11 = L3_exc_spikes(:,L3_exc_spikes_im10_idx:L3_exc_spikes_im11_idx);
L3_exc_spikes_im12_idx = find(L3_exc_spikes(2,:)>12,1);
L3_exc_spikes_im12 = L3_exc_spikes(:,L3_exc_spikes_im11_idx:L3_exc_spikes_im12_idx);
L3_exc_spikes_im13_idx = find(L3_exc_spikes(2,:)>13,1);
L3_exc_spikes_im13 = L3_exc_spikes(:,L3_exc_spikes_im12_idx:L3_exc_spikes_im13_idx);
L3_exc_spikes_im14_idx = find(L3_exc_spikes(2,:)>14,1);
L3_exc_spikes_im14 = L3_exc_spikes(:,L3_exc_spikes_im13_idx:L3_exc_spikes_im14_idx);
L3_exc_spikes_im15_idx = find(L3_exc_spikes(2,:)>15,1);
L3_exc_spikes_im15 = L3_exc_spikes(:,L3_exc_spikes_im14_idx:L3_exc_spikes_im15_idx);
L3_exc_spikes_im16_idx = find(L3_exc_spikes(2,:)>16,1);
L3_exc_spikes_im16 = L3_exc_spikes(:,L3_exc_spikes_im15_idx:L3_exc_spikes_im16_idx);

% calculate average firing rates for neurons for each image
[L3e_neurons_im1, L3e_count_im1, L3e_rates_im1] = rates(L3_exc_spikes_im1,4096,1);
[L3e_neurons_im2, L3e_count_im2, L3e_rates_im2] = rates(L3_exc_spikes_im2,4096,1);
[L3e_neurons_im3, L3e_count_im3, L3e_rates_im3] = rates(L3_exc_spikes_im3,4096,1);
[L3e_neurons_im4, L3e_count_im4, L3e_rates_im4] = rates(L3_exc_spikes_im4,4096,1);
[L3e_neurons_im5, L3e_count_im5, L3e_rates_im5] = rates(L3_exc_spikes_im5,4096,1);
[L3e_neurons_im6, L3e_count_im6, L3e_rates_im6] = rates(L3_exc_spikes_im6,4096,1);
[L3e_neurons_im7, L3e_count_im7, L3e_rates_im7] = rates(L3_exc_spikes_im7,4096,1);
[L3e_neurons_im8, L3e_count_im8, L3e_rates_im8] = rates(L3_exc_spikes_im8,4096,1);
[L3e_neurons_im9, L3e_count_im9, L3e_rates_im9] = rates(L3_exc_spikes_im9,4096,1);
[L3e_neurons_im10, L3e_count_im10, L3e_rates_im10] = rates(L3_exc_spikes_im10,4096,1);
[L3e_neurons_im11, L3e_count_im11, L3e_rates_im11] = rates(L3_exc_spikes_im11,4096,1);
[L3e_neurons_im12, L3e_count_im12, L3e_rates_im12] = rates(L3_exc_spikes_im12,4096,1);
[L3e_neurons_im13, L3e_count_im13, L3e_rates_im13] = rates(L3_exc_spikes_im13,4096,1);
[L3e_neurons_im14, L3e_count_im14, L3e_rates_im14] = rates(L3_exc_spikes_im14,4096,1);
[L3e_neurons_im15, L3e_count_im15, L3e_rates_im15] = rates(L3_exc_spikes_im15,4096,1);
[L3e_neurons_im16, L3e_count_im16, L3e_rates_im16] = rates(L3_exc_spikes_im16,4096,1);

% get average firing rates for specific image sets (see lab notebook)
L3e_rates_set1 = (L3e_rates_im3+L3e_rates_im4+L3e_rates_im16+L3e_rates_im6+L3e_rates_im5+L3e_rates_im15+L3e_rates_im10+L3e_rates_im1)/8;
L3e_rates_set2 = (L3e_rates_im3+L3e_rates_im4+L3e_rates_im16+L3e_rates_im6+L3e_rates_im5+L3e_rates_im15+L3e_rates_im8+L3e_rates_im11)/8;
L3e_rates_set3 = (L3e_rates_im3+L3e_rates_im4+L3e_rates_im16+L3e_rates_im6+L3e_rates_im8+L3e_rates_im11+L3e_rates_im14+L3e_rates_im13)/8;
L3e_rates_set4 = (L3e_rates_im3+L3e_rates_im4+L3e_rates_im8+L3e_rates_im11+L3e_rates_im14+L3e_rates_im13+L3e_rates_im12+L3e_rates_im2)/8;

neuron_num = 1079

figure()
hold on
bar([1:4],[L3e_rates_set1(neuron_num),L3e_rates_set2(neuron_num),L3e_rates_set3(neuron_num),L3e_rates_set4(neuron_num)])
title(['Average firing rate across stimuli for exc. LIF neuron #',num2str(neuron_num),' in layer 3'])
ylabel('average firing rate (Hz)')
xlabel('image set')
xticks([1:4])
grid on

%% plot distribution of rates

[L3_exc_neurons, L3_exc_count, L3_exc_rates] = rates(L3_exc_spikes,4096,16);

figure()
hist(L3_exc_rates,50)
xlabel('average firing rate (Hz)')
ylabel('number of neurons')
title('Distribution of average firing rates for L3 exc. LIF neurons')
grid on
%% plots

% plot spikes
figure()
scatter(L1_exc_spikes(2,:),L1_exc_spikes(1,:),0.5,'r')
title('layer 1 excitatory neurons')
xlabel('time (s)')
ylabel('neuron index')

% plot spike count
figure()
hist(L1e_count,25)
xlabel('average firing rate (Hz)')
ylabel('number of neuons')
grid on

% plot spike rate
figure()
hist(L1e_rates,25)
xlabel('average firing rate (Hz)')
ylabel('number of neuons')
grid on

% % plot weights
% figure()
% hist(L1_exc_L2_exc_weights(6,:),25)
% title('weights of synapses from layer 1 excitatory to layer 2 excitatory neurons')
% xlabel('synaptic weight (Siemens)')
% ylabel('number of neurons')
% grid('on')
% 
% % plot rates
% figure()
% hist(L1_exc_L2_exc_weights(2,:),100)
% title('firing rates of layer 1 excitatory neurons')
% xlabel('rate (Hz)')
% ylabel('number of neurons')
% grid('on')

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
