mod camera;
mod loader;
mod renderer;
mod simulation;

use bytemuck::{Pod, Zeroable};
use camera::{Camera, CameraController, CameraUniform};
use glam::{Mat4, Vec3};
use loader::Loader;
use renderer::{Renderer, Vertex};
use simulation::{NeuronConfig, Simulation};
use std::{f32::consts, sync::Arc, thread, time::Duration};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

struct State {
    window: Arc<Window>,
    loader: Loader,
    renderer: Renderer,
    camera_controller: CameraController,
    simulation: Simulation,
}

impl State {
    async fn new(window: Arc<Window>) -> State {
        let size = window.inner_size();
        let loader = Loader::new(size.width, size.height).unwrap();

        let mut renderer = Renderer::new(
            window.clone(),
            &loader.point_vertices,
            &loader.strong_indices,
            &loader.camera_uniform,
        )
        .await;

        let camera_controller = CameraController::new(100.0, 0.0005);

        let num_neurons = loader.point_vertices.len() as u32;
        let group_ids: Vec<u32> = loader.point_vertices.iter().map(|v| v.group_id).collect();

        let configs = vec![
            NeuronConfig {
                a: 0.02,
                b: 0.2,
                c: -65.0,
                d: 8.0,
            }, // ACH
            NeuronConfig {
                a: 0.1,
                b: 0.2,
                c: -65.0,
                d: 2.0,
            }, // GABA
            NeuronConfig {
                a: 0.02,
                b: 0.2,
                c: -65.0,
                d: 8.0,
            },
            // GLUT
            NeuronConfig {
                a: 0.02,
                b: 0.25,
                c: -65.0,
                d: 6.0,
            }, // DA
            NeuronConfig {
                a: 0.03,
                b: 0.25,
                c: -60.0,
                d: 4.0,
            }, // SER
            NeuronConfig {
                a: 0.02,
                b: 0.2,
                c: -50.0,
                d: 2.0,
            },
            // OCT
        ];

        let simulation = Simulation::new(
            &renderer.device,
            num_neurons,
            &group_ids,
            &configs,
            &loader.indices,
            &loader.weights,
            &loader.stimulated_neurons,
        )
        .await
        .unwrap();

        renderer.update_spike_bind_group(&simulation.neuron_states_buffer);

        Self {
            window,
            loader,
            renderer,
            camera_controller,
            simulation,
        }
    }

    fn get_window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.renderer.resize(new_size);
        self.loader.resize_camera_uniform(new_size);
        self.renderer.update_camera(&self.loader.camera_uniform);
    }

    fn update(&mut self) {
        self.simulation
            .step(&self.renderer.device, &self.renderer.queue);
        self.camera_controller
            .update_camera(&mut self.loader.camera);
        self.loader.resize_camera_uniform(self.renderer.size);
        self.renderer.update_camera(&self.loader.camera_uniform);
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("neurilium"))
                .unwrap(),
        );

        let state = pollster::block_on(State::new(window.clone()));
        self.state = Some(state);
        window.request_redraw();
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            state.get_window().request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = self.state.as_mut().unwrap();

        state
            .camera_controller
            .process_events(&state.window, &event);
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }

            WindowEvent::Resized(physical_size) => {
                state.resize(physical_size);
            }

            WindowEvent::RedrawRequested => {
                state.update();
                state.renderer.render();
            }

            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            if let DeviceEvent::MouseMotion { delta } = event {
                state
                    .camera_controller
                    .process_mouse(delta.0, delta.1, &mut state.loader.camera);

                state.camera_controller.center_cursor(&state.window);
            }
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App { state: None };
    event_loop.run_app(&mut app).unwrap();
}
