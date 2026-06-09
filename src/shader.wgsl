struct NeuronState {
    v: f32,
    u: f32,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    aspect: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<storage, read> neuron_states: array<NeuronState>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) group_id: u32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_point(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) i_idx: u32,
    @location(0) pos: vec3<f32>,
    @location(1) group_id: u32,
) -> VertexOutput {
    let corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0) 
    );

    var color = vec3<f32>(0.3, 0.3, 0.3);

    if (group_id == 1u) {
        color = vec3<f32>(0.760525, 0.760525, 0.760525); // ACH 
    } else if (group_id == 2u) {
        color = vec3<f32>(0.8, 0.8, 0.8); // GABA 
    } else if (group_id == 3u) {
        color = vec3<f32>(0.95, 0.95, 0.95); // GLUT 
    } else if (group_id == 4u) {
        color = vec3<f32>(0.5, 0.5, 0.9); // DA 
    } else if (group_id == 5u) {
        color = vec3<f32>(0.5, 0.5, 0.5); // SER 
    } else if (group_id == 6u) {
        color = vec3<f32>(0.0, 0.0, 0.0); // OCT
    }

    let state = neuron_states[i_idx];
    let v = state.v;

    var t = 0.0;
    if (v != 0.0) {
      t = saturate((v + 80.0) / (30.0 + 80.0));
    } 
    var final_color = color;
    if (t > 0.5) {
      final_color = vec3(0.830770, 0.973445, 0.171441);
    }



    let corner = corners[v_idx];
    let point_size = 100.0; 

    let world_pos = vec4<f32>(pos, 1.0);
    let clip_pos = camera.view_proj * world_pos;

    let pulse = 1.0 + (t * 4.0);
    let offset = vec2<f32>(corner.x / camera.aspect, corner.y) * point_size * pulse;

    var out: VertexOutput;
    out.clip_pos = vec4<f32>(clip_pos.xy + offset, clip_pos.zw);
    out.uv = corner * 0.5 + 0.5;
    out.color = final_color; 
    return out;
}

@fragment
fn fs_point(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv - vec2<f32>(0.5, 0.5));
    if (dist > 0.5) {
        discard;
    }
    return vec4<f32>(in.color, 1.0);
}


@vertex
fn vs_line(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let clip_pos = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.clip_pos = clip_pos;

    out.color = vec3<f32>(0.3, 0.3, 0.3);
    out.uv = vec2<f32>(0.0);
    return out;
}

@fragment
fn fs_line(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}


