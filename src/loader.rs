use crate::camera::{Camera, CameraUniform};
use crate::renderer::Vertex;
use glam::{Mat4, Vec3};
use serde::Deserialize;
use std::{
    collections::HashMap,
    f32::consts,
    fs::File,
    io::{BufReader, Error, ErrorKind, Result},
};
use winit::dpi::PhysicalSize;

#[derive(Debug, Deserialize)]
struct CsvRow {
    root_id: u64,
    position: String,
    supervoxel_id: u64,
}

#[derive(Debug, Deserialize)]
struct ConnectionRow {
    pre_root_id: u64,
    post_root_id: u64,
    syn_count: u32,
}

#[derive(Debug, Deserialize)]
struct NtRow {
    root_id: u64,
    group: String,
    nt_type: String,
}

#[derive(Clone)]
struct NeuronMeta {
    group: String,
    nt_type: String,
}

fn nt_to_id(nt: &str) -> u32 {
    match nt.to_uppercase().as_str() {
        "ACH" => 1,
        "GABA" => 2,
        "GLUT" => 3,
        "DA" => 4,
        "SER" => 5,
        "OCT" => 6,
        _ => 0,
    }
}

pub struct Loader {
    pub point_vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub weights: Vec<u32>,
    pub stimulated_neurons: Vec<u32>,
    pub strong_indices: Vec<u32>,

    pub camera: Camera,
    pub camera_uniform: CameraUniform,
}

impl Loader {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        println!("Loading neurons...");
        let nt_file = File::open("data/neurons.csv")?;
        let mut nt_rdr = csv::Reader::from_reader(BufReader::new(nt_file));
        let mut nt_map = HashMap::new();

        for result in nt_rdr.deserialize() {
            let row: NtRow = result.map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            nt_map.insert(
                row.root_id,
                NeuronMeta {
                    group: row.group,
                    nt_type: row.nt_type,
                },
            );
        }
        println!("Loaded neurons.");

        println!("Loading neuron positions...");
        let csv_file = File::open("data/coordinates.csv")?;
        let mut rdr = csv::Reader::from_reader(BufReader::new(csv_file));

        let mut raw_points = Vec::new();
        let mut id_to_index = HashMap::new();

        for result in rdr.deserialize() {
            let row: CsvRow = result.map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

            if nt_map.contains_key(&row.root_id) && !id_to_index.contains_key(&row.root_id) {
                let pos = Self::parse_position(&row.position)
                    .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

                let vertex_idx = raw_points.len() as u32;
                id_to_index.insert(row.root_id, vertex_idx);
                raw_points.push((row.root_id, pos));
            }
        }

        let count = raw_points.len();
        if count == 0 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "No neurons found in CSV file.",
            ));
        }

        println!("Loaded neuron positions.");

        println!("Loading neuron connections...");
        let conn_file = File::open("data/connections_princeton.csv")?;
        let mut conn_rdr = csv::Reader::from_reader(BufReader::new(conn_file));
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        let mut strong_indices = Vec::new();

        let mut kept_connections = 0;

        for result in conn_rdr.deserialize() {
            let row: ConnectionRow = result.map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

            if let (Some(&pre_idx), Some(&post_idx)) = (
                id_to_index.get(&row.pre_root_id),
                id_to_index.get(&row.post_root_id),
            ) {
                indices.push(pre_idx);
                indices.push(post_idx);
                weights.push(row.syn_count);
                kept_connections += 1;

                if kept_connections % 5000 == 0 {
                    strong_indices.push(pre_idx);
                    strong_indices.push(post_idx);
                }
            }
        }

        println!("Loaded neuron connections.");

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_z = 0.0;
        for (_, pos) in &raw_points {
            sum_x += pos[0];
            sum_y += pos[1];
            sum_z += pos[2];
        }
        let center = [
            sum_x / count as f32,
            sum_y / count as f32,
            sum_z / count as f32,
        ];

        let scale_factor = 0.05f32;
        let mut point_vertices = Vec::with_capacity(count);
        let mut stimulated_neurons = Vec::new();

        for (idx, (root_id, raw_pos)) in raw_points.into_iter().enumerate() {
            let meta = nt_map.get(&root_id).cloned().unwrap_or_else(|| NeuronMeta {
                group: "UNKNOWN".to_string(),
                nt_type: "UNKNOWN".to_string(),
            });

            let group_id = nt_to_id(&meta.nt_type);

            if meta.group == "ME" {
                stimulated_neurons.push(idx as u32);
            }

            let render_pos = [
                (raw_pos[0] - center[0]) * scale_factor,
                -(raw_pos[1] - center[1]) * scale_factor,
                (raw_pos[2] - center[2]) * scale_factor,
            ];

            point_vertices.push(Vertex {
                position: render_pos,
                group_id,
                source_pos: [0.0; 3],
            });
        }
        println!("Loaded brain of {} neurons.", point_vertices.len());

        let camera = Camera {
            eye: Vec3::new(0.0, 0.0, -50000.0),
            yaw: consts::FRAC_PI_2,
            pitch: 0.0,
        };

        let mut camera_uniform = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            aspect: width as f32 / height as f32,
            _padding: [0.0; 3],
        };

        camera_uniform.update_view_proj(&camera, camera_uniform.aspect);

        Ok(Self {
            point_vertices,
            indices,
            weights,
            stimulated_neurons,
            strong_indices,
            camera,
            camera_uniform,
        })
    }

    fn parse_position(pos_str: &str) -> std::result::Result<[f32; 3], String> {
        let clean = pos_str.trim().trim_start_matches('[').trim_end_matches(']');

        let parts: Vec<&str> = clean.split_whitespace().collect();
        if parts.len() != 3 {
            return Err(format!("Expected 3 coordinates, got {}", parts.len()));
        }

        let x = parts[0].parse::<f32>().map_err(|e| e.to_string())?;
        let y = parts[1].parse::<f32>().map_err(|e| e.to_string())?;
        let z = parts[2].parse::<f32>().map_err(|e| e.to_string())?;

        Ok([x, y, z])
    }

    pub fn resize_camera_uniform(&mut self, new_size: PhysicalSize<u32>) {
        let aspect = new_size.width as f32 / new_size.height as f32;
        self.camera_uniform.view_proj =
            self.camera.build_view_projection(aspect).to_cols_array_2d();
        self.camera_uniform.aspect = aspect;
    }
}
