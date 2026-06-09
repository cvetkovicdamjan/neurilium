use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Error, ErrorKind, Read, Result};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NeuronState {
    pub v: f32,
    pub u: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Synapse {
    pub synapse_target: u32,
    pub weight: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NeuronConfig {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
}

pub struct Simulation {
    pub neuron_states_buffer: wgpu::Buffer,
    pub synaptic_inputs_buffer: wgpu::Buffer,
    pub spikes_buffer: wgpu::Buffer,
    pub synapses_buffer: wgpu::Buffer,
    pub configs_buffer: wgpu::Buffer,
    pub synapse_offsets_buffer: wgpu::Buffer,
    pub external_currents_buffer: wgpu::Buffer,
    pub neuron_groups_buffer: wgpu::Buffer,

    pub compute_pipeline_update: wgpu::ComputePipeline,
    pub compute_pipeline_propagate: wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,

    pub external_currents: Vec<f32>,
    pub num_neurons: u32,
    pub stimulated_neurons: Vec<u32>,
}

impl Simulation {
    pub async fn new(
        device: &wgpu::Device,
        num_neurons: u32,
        neuron_groups: &[u32],
        configs: &[NeuronConfig],
        indices: &[u32],
        weights: &[u32],
        stimulated_neurons: &[u32],
    ) -> Result<Self> {
        let initial_states = vec![NeuronState { v: -65.0, u: -13.0 }; num_neurons as usize];
        let initial_synaptic = vec![0i32; num_neurons as usize];
        let initial_spikes = vec![0u32; num_neurons as usize];

        let mut source_to_synapses = vec![Vec::new(); num_neurons as usize];
        for (i, chunk) in indices.chunks_exact(2).enumerate() {
            let source = chunk[0] as usize;
            let target = chunk[1] as usize;
            let weight = weights[i];
            if source < num_neurons as usize && target < num_neurons as usize {
                source_to_synapses[source].push(Synapse {
                    synapse_target: target as u32,
                    weight,
                });
            }
        }

        let mut synapses = Vec::new();
        let mut synapse_offsets = Vec::with_capacity(num_neurons as usize + 1);
        synapse_offsets.push(0u32);

        for syns in &source_to_synapses {
            for &syn in syns {
                synapses.push(syn);
            }
            synapse_offsets.push(synapses.len() as u32);
        }

        let external_currents = vec![0.0f32; num_neurons as usize];

        let neuron_states_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Neuron States Buffer"),
            contents: bytemuck::cast_slice(&initial_states),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let synaptic_inputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Synaptic Inputs Buffer"),
            contents: bytemuck::cast_slice(&initial_synaptic),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let spikes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Spikes Buffer"),
            contents: bytemuck::cast_slice(&initial_spikes),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let synapses_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Synapses Buffer"),
            contents: bytemuck::cast_slice(&synapses),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let configs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Configs Buffer"),
            contents: bytemuck::cast_slice(configs),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let synapse_offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Synapse Offsets Buffer"),
            contents: bytemuck::cast_slice(&synapse_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let external_currents_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Simulation External Currents Buffer"),
                contents: bytemuck::cast_slice(&external_currents),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let neuron_groups_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Group IDs Buffer"),
            contents: bytemuck::cast_slice(neuron_groups),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("compute.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Simulation Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Simulation Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: neuron_states_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: synaptic_inputs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spikes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: synapses_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: configs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: synapse_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: external_currents_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: neuron_groups_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Simulation Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            ..Default::default()
        });

        let compute_pipeline_update =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Simulation Update Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("update_neurons"),
                compilation_options: Default::default(),
                cache: None,
            });

        let compute_pipeline_propagate =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Simulation Propagate Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("propagate_spikes"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            neuron_states_buffer,
            synaptic_inputs_buffer,
            spikes_buffer,
            synapses_buffer,
            configs_buffer,
            synapse_offsets_buffer,
            external_currents_buffer,
            neuron_groups_buffer,
            compute_pipeline_update,
            compute_pipeline_propagate,
            bind_group,
            external_currents,
            num_neurons,
            stimulated_neurons: stimulated_neurons.to_vec(),
        })
    }

    pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.external_currents.fill(0.0);

        let injection_current = 5.0f32;
        for &neuron_idx in &self.stimulated_neurons {
            if (neuron_idx as usize) < self.external_currents.len() {
                self.external_currents[neuron_idx as usize] = injection_current;
            }
        }

        queue.write_buffer(
            &self.external_currents_buffer,
            0,
            bytemuck::cast_slice(&self.external_currents),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Simulation Step Encoder"),
        });

        let workgroup_count = (self.num_neurons + 63) / 64;

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Neuron Equations Pass"),
                ..Default::default()
            });
            compute_pass.set_pipeline(&self.compute_pipeline_update);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Spike Propagation Pass"),
                ..Default::default()
            });
            compute_pass.set_pipeline(&self.compute_pipeline_propagate);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        queue.submit([encoder.finish()]);
    }
}
