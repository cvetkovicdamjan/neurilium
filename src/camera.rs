use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::f32::consts;
use winit::{
    dpi::{LogicalPosition, PhysicalPosition},
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window},
};

pub struct Camera {
    pub eye: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn build_view_projection(&self, aspect: f32) -> Mat4 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let forward = Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize();
        let view = Mat4::look_at_rh(self.eye, self.eye + forward, Vec3::Y);
        let proj = Mat4::perspective_rh(consts::FRAC_PI_4, aspect, 0.1, 1000000.0);
        proj * view
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub aspect: f32,
    pub _padding: [f32; 3],
}

impl CameraUniform {
    pub fn update_view_proj(&mut self, camera: &Camera, aspect: f32) {
        self.view_proj = camera.build_view_projection(aspect).to_cols_array_2d();
        self.aspect = aspect;
    }
}

pub struct CameraController {
    speed: f32,
    sensitivity: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    pub is_focused: bool,
    initial_cursor_pos: Option<LogicalPosition<f64>>,
    pub mouse_pos: PhysicalPosition<f64>,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            is_focused: false,
            initial_cursor_pos: None,
            mouse_pos: PhysicalPosition::new(0.0, 0.0),
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64, camera: &mut Camera) {
        if self.is_focused {
            camera.yaw += mouse_dx as f32 * self.sensitivity;
            camera.pitch -= mouse_dy as f32 * self.sensitivity;

            let limit = 89.0f32.to_radians();
            if camera.pitch > limit {
                camera.pitch = limit;
            } else if camera.pitch < -limit {
                camera.pitch = -limit;
            }
        }
    }

    pub fn handle_mouse_capture(&mut self, window: &Window, captured: bool) {
        self.is_focused = captured;
        window.set_cursor_visible(!captured);
        let mode = if captured {
            CursorGrabMode::Confined
        } else {
            CursorGrabMode::None
        };
        let _ = window.set_cursor_grab(mode);
    }

    pub fn center_cursor(&self, window: &Window) {
        if self.is_focused {
            let size = window.inner_size();
            let center = LogicalPosition::new(size.width as f32 / 2.0, size.height as f32 / 2.0);
            let _ = window.set_cursor_position(center);
        }
    }

    pub fn process_events(&mut self, window: &Window, event: &WindowEvent) {
        match event {
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => match state {
                ElementState::Pressed => {
                    if !self.is_focused {
                        self.handle_mouse_capture(window, true);
                    }
                }
                ElementState::Released => {
                    if self.is_focused {
                        self.handle_mouse_capture(window, false);

                        if let Some(pos) = self.initial_cursor_pos {
                            let _ = window.set_cursor_position(pos);
                        }
                    }
                }
            },

            WindowEvent::CursorMoved { position, .. } => {
                if !self.is_focused {
                    self.initial_cursor_pos = Some(LogicalPosition::new(position.x, position.y));
                    self.mouse_pos = PhysicalPosition::new(position.x, position.y);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let is_pressed = event.state.is_pressed();
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => {
                            self.is_forward_pressed = is_pressed;
                        }
                        KeyCode::KeyS => {
                            self.is_backward_pressed = is_pressed;
                        }
                        KeyCode::KeyA => {
                            self.is_left_pressed = is_pressed;
                        }
                        KeyCode::KeyD => {
                            self.is_right_pressed = is_pressed;
                        }
                        KeyCode::KeyE => {
                            self.is_up_pressed = is_pressed;
                        }
                        KeyCode::KeyQ => {
                            self.is_down_pressed = is_pressed;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        let forward = Vec3::new(camera.yaw.cos(), 0.0, camera.yaw.sin()).normalize();
        let right = Vec3::new(-camera.yaw.sin(), 0.0, camera.yaw.cos()).normalize();

        if self.is_focused {
            if self.is_forward_pressed {
                camera.eye += forward * self.speed;
            }
            if self.is_backward_pressed {
                camera.eye -= forward * self.speed;
            }
            if self.is_right_pressed {
                camera.eye += right * self.speed;
            }
            if self.is_left_pressed {
                camera.eye -= right * self.speed;
            }
            if self.is_up_pressed {
                camera.eye += Vec3::Y * self.speed;
            }
            if self.is_down_pressed {
                camera.eye -= Vec3::Y * self.speed;
            }
        }
    }
}
