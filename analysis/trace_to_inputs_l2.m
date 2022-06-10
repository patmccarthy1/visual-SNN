function inputs = trace_to_inputs_l2(neuron_num,weight,layer_1_exc_layer_2_exc_weights_8_0s)
    % find synapses from previous layer neurons to this neuron
    syn_idx = find(layer_1_exc_layer_2_exc_weights_8_0s(2,:) == neuron_num);
    inputs = [syn_idx; layer_1_exc_layer_2_exc_weights_8_0s(1,syn_idx);layer_1_exc_layer_2_exc_weights_8_0s(2,syn_idx);layer_1_exc_layer_2_exc_weights_8_0s(3,syn_idx)+weight];
    % sort by weights in descending order 
    inputs = transpose(inputs);
    inputs = sortrows(inputs,4,'descend');
    inputs = transpose(inputs);
end
