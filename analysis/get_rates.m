function [neurons, N_spikes, the_rates] = get_rates(spks,N_neurons,simulation_time)
    idx = spks(1,:)+1;
    neurons = [1:N_neurons];
    N_spikes = zeros(1,length(neurons));
    for i=idx
        N_spikes(1,i) = N_spikes(1,i)+1;
    end
    for i=neurons
        the_rates(i) = N_spikes(i)/simulation_time;
    end
end