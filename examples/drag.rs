extern crate line_2d;
extern crate prototty_unix;

use line_2d::{Coord, DirectedLineSegment};
use prototty_unix::prototty_input::{inputs, Input};
use prototty_unix::prototty_render::{Rgb24, View, ViewCell, ViewGrid};
use prototty_unix::Context;

struct AppView;
#[derive(Default)]
struct App {
    coord: Option<Coord>,
    last_clicked_coord: Option<Coord>,
}

impl View<App> for AppView {
    fn view<G: ViewGrid>(&mut self, app: &App, offset: Coord, depth: i32, grid: &mut G) {
        let white = Rgb24::new(255, 255, 255);
        match (app.last_clicked_coord, app.coord) {
            (Some(last_clicked_coord), Some(coord)) => {
                for coord in DirectedLineSegment::new(last_clicked_coord, coord) {
                    grid.set_cell(
                        coord + offset,
                        depth,
                        ViewCell::new().with_character(' ').with_background(white),
                    );
                }
            }
            _ => (),
        }
    }
}

fn main() {
    let mut context = Context::new().unwrap();
    let mut app = App::default();
    loop {
        match context.wait_input().unwrap() {
            inputs::ESCAPE | inputs::ETX => break,
            Input::MouseMove(coord) => {
                app.coord = Some(coord);
            }
            Input::MousePress { coord, button: _ } => {
                app.last_clicked_coord = Some(coord);
            }
            Input::MouseRelease {
                coord: _,
                button: _,
            } => {
                app.last_clicked_coord = None;
            }
            _ => (),
        }
        context.render(&mut AppView, &app).unwrap();
    }
}
