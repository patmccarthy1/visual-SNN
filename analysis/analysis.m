%%  Analysis of spike and weight data to identify feature selective neurons
clc; close all; clear all

% % parameters
N_images = 16;
show_time = 1;                                                % length of time stimulus is presented for (in seconds)
period = N_images*show_time;         % length of an epoch (in seconds)
N_epochs = 20+1;                                          % add 1 for test epoch

%% read spike data and isolate by epoch and image stimulus 

spikes_names = {'L0';'L1e';'L1i';'L2e';'L2i';'L3e';'L3i';'L4e';'L4i'}  % names 

% layer 0 spikes
L0_spikes = importdata('data/layer_0_full_spikes.csv');
L0_spikes_epoch_idx{1} = 1;
L0_spikes_im_idx{1} = 1;
for epoch = 1:N_epochs % loop through epochs and isolate spikes for given epoch
    L0_spikes_epoch_idx{epoch+1} = find(L0_spikes(2,:)>period*epoch,1);
    L0_spikes_epoch = L0_spikes(:,L0_spikes_epoch_idx{epoch}:L0_spikes_epoch_idx{epoch+1});
    size(L0_spikes_epoch)
    for im = 1:N_images % loop through images and isolate spikes for given image within epoch
        L0_spikes_im_idx{im+1} = find(L0_spikes_epoch(2,:)>(period*(epoch-1) + show_time*im),1)
        L0_spikes_im = L0_spikes_epoch(:,L0_spikes_im_idx{im}:L0_spikes_im_idx{im+1});
        spikes{1,epoch,im} = L0_spikes_im; % store spikes for L0 for given epoch and image
    end
end

% layers 1-4 exc and inh spikes
for i = 1:4 % loop through layers 1-4

    % excitatory layer
    exc_spikes = importdata(['data/layer_' num2str(i) '_excitatory_full_spikes.csv']);
    exc_spikes_epoch_idx{1} = 1;
    exc_spikes_im_idx{1} = 1;
    for epoch = 1:N_epochs % loop through epochs and isolate spikes for given epoch
        exc_spikes_epoch_idx{epoch+1} = find(exc_spikes(2,:)>period*epoch,1);
        exc_spikes_epoch = exc_spikes(:,exc_spikes_epoch_idx{epoch}:exc_spikes_epoch_idx{epoch+1});
        for im = 1:N_images % loop through images and isolate spikes for given image within epoch
            exc_spikes_im_idx{im+1} = find(exc_spikes_epoch(2,:)>(period*epoch + show_time*im),1);
            exc_spikes_im = exc_spikes_epoch(:,exc_spikes_im_idx{im}:exc_spikes_im_idx{im+1});
            spikes{2*i,epoch,im} = exc_spikes_im; % store spikes for L0 for given epoch and image
        end
    end
    
    % inhibitory layer
    inh_spikes = importdata(['data/layer_' num2str(i) '_inhibitory_full_spikes.csv']);
    inh_spikes_epoch_idx{1} = 1;
    inh_spikes_im_idx{1} = 1;
    for epoch = 1:N_epochs % loop through epochs and isolate spikes for given epoch
        inh_spikes_epoch_idx{epoch+1} = find(inh_spikes(2,:)>period*epoch,1);
        inh_spikes_epoch = inh_spikes(:,inh_spikes_epoch_idx{epoch}:inh_spikes_epoch_idx{epoch+1});
        for im = 1:N_images % loop through images and isolate spikes for given image within epoch
            inh_spikes_im_idx{im+1} = find(inh_spikes_epoch(2,:)>(period*epoch + show_time*im),1);
            inh_spikes_im = inh_spikes_epoch(:,inh_spikes_im_idx{im}:inh_spikes_im_idx{im+1});
            spikes{(2*i)+1,epoch,im} = inh_spikes_im; % store spikes for L0 for given epoch and image
        end
    end
    
end

%% read weights data

% layer 0 to 1 exc
for epoch = 1:N_epochs-1 % loop through epochs (but do not include test epoch)
    x = importdata(['data/layer_0_layer_1_exc_weights_8s_idx_pre_epoch_' num2str(epoch-1) '.csv']);
    idx_pre = importdata(['data/layer_0_layer_1_exc_weights_8s_idx_pre_epoch_' num2str(epoch-1) '.csv']);
    x = importdata(['data/layer_0_layer_1_exc_weights_8s_x_pre_epoch_' num2str(epoch-1) '.csv']);
    y = importdata(['data/layer_0_layer_1_exc_weights_8s_y_pre_epoch_' num2str(epoch-1) '.csv']);
    idx_post = importdata(['data/layer_0_layer_1_exc_weights_8s_idx_post_epoch_' num2str(epoch-1) '.csv']);
    f = importdata(['data/layer_0_layer_1_exc_weights_8s_f_pre_epoch_' num2str(epoch-1) '.csv']);
    w = importdata(['data/layer_0_layer_1_exc_weights_8s_w_epoch_' num2str(epoch-1) '.csv']);
    weights{1,epoch} = [idx_pre; x; y; idx_post; f; w];
end


% rest of layers
for i = 1:3 % loop through layers
    for epoch = 1:N_epochs-1 % loop through epochs (but do not include test epoch)
        idx_pre = importdata(['data/layer_' num2str(i) 'exc_layer_' num2str(i+1) 'exc_weights_8s_idx_pre_epoch_' num2str(epoch-1) '.csv']);
        idx_post = importdata(['data/layer_' num2str(i) 'exc_layer_' num2str(i+1) 'exc_weights_8s_idx_post_epoch_' num2str(epoch-1) '.csv']);
        w = importdata(['data/layer_' num2str(i) 'exc_layer_' num2str(i+1) 'exc_weights_8s_w_epoch_' num2str(epoch-1) '.csv']);
        weights{i+1,epoch} = [idx_pre; idx_post; w];
    end
end

%% read spike data and isolate by epoch and image stimulus 

spikes_names = {'L0';'L1e';'L1i';'L2e';'L2i';'L3e';'L3i';'L4e';'L4i'}  % names 

% layer 0 spikes
L0_spikes = importdata('data/layer_0_full_spikes.csv');
L0_spikes_epoch_idx{1} = 1;
L0_spikes_im_idx{1} = 1;
for epoch = 1:N_epochs % loop through epochs and isolate spikes for given epoch
    L0_spikes_epoch_idx{epoch+1} = find(L0_spikes(2,:)>period*epoch,1);
    L0_spikes_epoch = L0_spikes(:,L0_spikes_epoch_idx{epoch}:L0_spikes_epoch_idx{epoch+1});
    for im = 1:N_images % loop through images and isolate spikes for given image within epoch
        L0_spikes_im_idx{epoch+1} = find(L0_spikes_epoch(2,:)>(period*epoch + show_time*im),1);
        L0_spikes_im = L0_spikes_epoch(:,L0_spikes_im_idx{epoch}:L0_spikes_im_idx{epoch+1});
        spikes{1,epoch,im} = L0_spikes_im; % store spikes for L0 for given epoch and image
    end
end

% layers 1-4 exc and inh spikes
for i = 1:4 % loop through layers 1-4

    % excitatory layer
    exc_spikes = importdata(['data/layer_' num2str(i) '_excitatory_full_spikes.csv']);
    exc_spikes_epoch_idx{1} = 1;
    exc_spikes_im_idx{1} = 1;
    for epoch = 1:N_epochs % loop through epochs and isolate spikes for given epoch
        exc_spikes_epoch_idx{epoch+1} = find(exc_spikes(2,:)>period*epoch,1);
        exc_spikes_epoch = exc_spikes(:,exc_spikes_epoch_idx{epoch}:exc_spikes_epoch_idx{epoch+1});
        for im = 1:N_images % loop through images and isolate spikes for given image within epoch
            exc_spikes_im_idx{epoch+1} = find(exc_spikes_epoch(2,:)>(period*epoch + show_time*im),1);
            exc_spikes_im = exc_spikes_epoch(:,exc_spikes_im_idx{epoch}:exc_spikes_im_idx{epoch+1});
            spikes{2*i,epoch,im} = exc_spikes_im; % store spikes for L0 for given epoch and image
        end
    end
    
    % inhibitory layer
    inh_spikes = importdata(['data/layer_' num2str(i) '_inhibitory_full_spikes.csv']);
    inh_spikes_epoch_idx{1} = 1;
    inh_spikes_im_idx{1} = 1;
    for epoch = 1:N_epochs % loop through epochs and isolate spikes for given epoch
        inh_spikes_epoch_idx{epoch+1} = find(inh_spikes(2,:)>period*epoch,1);
        inh_spikes_epoch = inh_spikes(:,inh_spikes_epoch_idx{epoch}:inh_spikes_epoch_idx{epoch+1});
        for im = 1:N_images % loop through images and isolate spikes for given image within epoch
            inh_spikes_im_idx{epoch+1} = find(inh_spikes_epoch(2,:)>(period*epoch + show_time*im),1);
            inh_spikes_im = inh_spikes_epoch(:,inh_spikes_im_idx{epoch}:inh_spikes_im_idx{epoch+1});
            spikes{(2*i)+1,epoch,im} = inh_spikes_im; % store spikes for L0 for given epoch and image
        end
    end
    
end

%% calculate average firing rates for neurons for each image
N_neurons = [65536,4096,1024,4096,1024,4096,1024,4096,1024] % number of neurons in each layer
for layer = 1:9
    for epoch = 1:N_epochs
        for im = 1:N_images
            [nrns,count,rates] = rates(spikes{layer,epoch,im},N_neurons(layer),1)
            rates{layer,epoch,im} = {nrns;count;rates};
        end
    end
end

%% raster plots
figure()
scatter(L3_exc_spikes_test(2,:)*1000,L3_exc_spikes_test(1,:),'*','MarkerEdgeColor','r')
xlabel('time (ms)')
ylabel('neuron index')
title('Raster plot for layer 3 excitatory neurons')
grid on

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

%% plot average firing rates
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
neurons = 727;
for l3_neuron = neurons
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


%%  function definitions

% function to calculate firing rates in given interval (as well as spike count) 
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
 
 % function to create  logical time series to represent a given neuron''s spikes
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


 % functions to trace pathways to inputs in previous layers
 
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
 
function inputs = trace_to_inputs_l3(neuron_num,layer_2_exc_layer_3_exc_weights_8_0s)
    % find synapses from previous layer neurons to this neuron
    syn_idx = find(layer_2_exc_layer_3_exc_weights_8_0s(2,:) == neuron_num);
    % store index, pre and post indices and weight
    inputs = [syn_idx;layer_2_exc_layer_3_exc_weights_8_0s(:,syn_idx)];
    % sort by weights in descending order 
    inputs = transpose(inputs);
    inputs = sortrows(inputs,4,'descend');
    inputs = transpose(inputs);
end
 
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

