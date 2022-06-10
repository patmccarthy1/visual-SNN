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