struct NeuronState {
  v: f32,
  u: f32,
}

struct Synapse {
  synapse_target: u32,
  weight: u32,
}


struct NeuronConfig {
  a: f32,
  b: f32,
  c: f32,
  d: f32,
}

struct NeuronConfigs {
  data: array<NeuronConfig, 6>,
}

const TYPE_ACH: u32 = 0u; 
const TYPE_GABA: u32 = 1u;  
const TYPE_GLUT: u32 = 2u; 
const TYPE_DA: u32 = 3u; 
const TYPE_SER: u32 = 4u;
const TYPE_OCT: u32 = 5u;


@group(0) @binding(0) var<storage, read_write> neuron_states: array<NeuronState>;
@group(0) @binding(1) var<storage, read_write> synaptic_inputs: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(3) var<storage, read> synapses: array<Synapse>;
@group(0) @binding(4) var<uniform> configs: NeuronConfigs;
@group(0) @binding(5) var<storage, read> synapse_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> external_currents: array<f32>;
@group(0) @binding(7) var<storage, read> neuron_groups: array<u32>;


@compute @workgroup_size(64)
fn update_neurons(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let num_neurons = arrayLength(&neuron_states);
    if (i >= num_neurons) {
        return;
    }

    var state = neuron_states[i];
    var v = state.v;
    var u = state.u;

    let synaptic_input_int = atomicExchange(&synaptic_inputs[i], 0);
    let synaptic_input_val = f32(synaptic_input_int);

    let i_current = external_currents[i];
    let group_id = neuron_groups[i];
    let config = configs.data[group_id];

    let dt = 0.2;
    v += dt * (0.04 * v * v + 5.0 * v + 140.0 - u + i_current + synaptic_input_val);
    u += dt * (config.a * (config.b * v - u));

    var spiked = 0u;
    if (v >= 30.0) {
        v = config.c;
        u += config.d;
        spiked = 1u;
    }

    neuron_states[i].v = v;
    neuron_states[i].u = u;
    spikes[i] = spiked;
}

@compute @workgroup_size(64)
fn propagate_spikes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let num_neurons = arrayLength(&neuron_states);
    if (i >= num_neurons) {
        return;
    }

    if (spikes[i] == 1u) {
	let source_type = neuron_groups[i];
        let start = synapse_offsets[i];
        let end = synapse_offsets[i + 1];

	if (source_type == TYPE_DA) {
            atomicAdd(&synaptic_inputs[i], 15);
        }

        for (var idx = start; idx < end; idx = idx + 1u) {
	    let synapse = synapses[idx];
            let synapse_target = synapse.synapse_target;
            let weight = synapse.weight;

	    switch (source_type) {
	      case TYPE_ACH: {
		atomicAdd(&synaptic_inputs[synapse_target], i32(weight));
	      }
	      case TYPE_GABA: {
	        atomicAdd(&synaptic_inputs[synapse_target], -i32(weight));
	      }
	      case TYPE_GLUT: {
		atomicAdd(&synaptic_inputs[synapse_target], i32(weight) / 2);
	      }
	      case TYPE_SER: {
		atomicAdd(&synaptic_inputs[synapse_target], -i32(weight) / 2);
	      }
	      case TYPE_OCT: {
		 atomicAdd(&synaptic_inputs[synapse_target], i32(weight) * 3 / 2);
	      }
	      default: {
		atomicAdd(&synaptic_inputs[synapse_target], i32(weight));
	      }

	   }
        }
    }
}


