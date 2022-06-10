function inputs = trace_to_inputs_l1(neuron_num,weight,layer_0_layer_1_exc_weights_8_0s)
    % find synapses from previous layer neurons to this neuron
    syn_idx = find(layer_0_layer_1_exc_weights_8_0s(4,:) == neuron_num);
    % store index, weight and x and y locations and filter number of all connected neurons
   inputs = [syn_idx;layer_0_layer_1_exc_weights_8_0s(6,syn_idx)+weight;layer_0_layer_1_exc_weights_8_0s(2,syn_idx);layer_0_layer_1_exc_weights_8_0s(3,syn_idx);layer_0_layer_1_exc_weights_8_0s(5,syn_idx)];
end