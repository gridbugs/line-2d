extern crate coord_2d;
use coord_2d::Axis;
pub use coord_2d::Coord;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DirectedLineSegment {
    pub start: Coord,
    pub end: Coord,
}

impl DirectedLineSegment {
    pub fn new(start: Coord, end: Coord) -> Self {
        Self { start, end }
    }
    pub fn delta(&self) -> Coord {
        self.end - self.start
    }
    pub fn iter(&self) -> DirectedLineSegmentIter {
        DirectedLineSegmentIter::new(*self)
    }
    fn steps(&self) -> Steps {
        Steps::new(self.delta())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Steps {
    major_axis: Axis,
    major_sign: i8,
    minor_sign: i8,
    accumulator: i64,
    major_delta_abs: u32,
    minor_delta_abs: u32,
}

impl Steps {
    fn new(delta: Coord) -> Self {
        let (major_axis, minor_axis) = if delta.x.abs() > delta.y.abs() {
            (Axis::X, Axis::Y)
        } else {
            (Axis::Y, Axis::X)
        };
        let major_sign = if delta.get(major_axis) < 0 { -1 } else { 1 };
        let minor_sign = if delta.get(minor_axis) < 0 { -1 } else { 1 };
        let (major_delta_abs, minor_delta_abs) = if delta == Coord::new(0, 0) {
            (1, 1)
        } else {
            (
                delta.get(major_axis).abs() as u32,
                delta.get(minor_axis).abs() as u32,
            )
        };
        let accumulator = 0;
        Self {
            major_axis,
            major_sign,
            minor_sign,
            accumulator,
            major_delta_abs,
            minor_delta_abs,
        }
    }
    fn prev(&mut self) -> Coord {
        self.accumulator -= self.minor_delta_abs as i64;
        if self.accumulator <= (self.major_delta_abs as i64 / 2) - self.major_delta_abs as i64 {
            self.accumulator += self.major_delta_abs as i64;
            Coord::new_axis(
                -self.major_sign as i32,
                -self.minor_sign as i32,
                self.major_axis,
            )
        } else {
            Coord::new_axis(-self.major_sign as i32, 0, self.major_axis)
        }
    }
    fn next(&mut self) -> Coord {
        self.accumulator += self.minor_delta_abs as i64;
        if self.accumulator > self.major_delta_abs as i64 / 2 {
            self.accumulator -= self.major_delta_abs as i64;
            Coord::new_axis(
                self.major_sign as i32,
                self.minor_sign as i32,
                self.major_axis,
            )
        } else {
            Coord::new_axis(self.major_sign as i32, 0, self.major_axis)
        }
    }
}

#[derive(Debug)]
pub struct DirectedLineSegmentIter {
    steps: Steps,
    directed_line_segment: DirectedLineSegment,
    current_coord: Coord,
}

impl DirectedLineSegmentIter {
    fn new(directed_line_segment: DirectedLineSegment) -> Self {
        let mut steps = directed_line_segment.steps();
        let backwards_step = steps.prev();
        let current_coord = directed_line_segment.start + backwards_step;
        Self {
            steps,
            directed_line_segment,
            current_coord,
        }
    }
}

impl Iterator for DirectedLineSegmentIter {
    type Item = Coord;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_coord == self.directed_line_segment.end {
            return None;
        }
        let step = self.steps.next();
        self.current_coord += step;
        Some(self.current_coord)
    }
}

impl IntoIterator for DirectedLineSegment {
    type Item = Coord;
    type IntoIter = DirectedLineSegmentIter;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod test {
    extern crate rand;
    use self::rand::rngs::StdRng;
    use self::rand::{Rng, SeedableRng};
    use super::*;

    fn manhatten_length(delta: Coord) -> usize {
        delta.x.abs().max(delta.y.abs()) as usize
    }

    fn test_properties(directed_line_segment: DirectedLineSegment) {
        let coords: Vec<_> = directed_line_segment.iter().collect();
        assert_eq!(
            coords.len(),
            manhatten_length(directed_line_segment.delta()) + 1
        );
        assert_eq!(*coords.first().unwrap(), directed_line_segment.start);
        assert_eq!(*coords.last().unwrap(), directed_line_segment.end);
        let mut steps = directed_line_segment.steps();
        for _ in 0..manhatten_length(directed_line_segment.delta()) {
            let before = steps.clone();
            steps.next();
            let mut after = steps.clone();
            after.prev();
            assert_eq!(
                before, after,
                "\n{:#?}\n{:#?}\n{:#?}",
                before, after, directed_line_segment
            );
        }
    }

    fn rand_int<R: Rng>(rng: &mut R) -> i32 {
        const MAX: i32 = 100;
        rng.gen::<i32>() % MAX
    }

    fn rand_coord<R: Rng>(rng: &mut R) -> Coord {
        Coord::new(rand_int(rng), rand_int(rng))
    }

    fn rand_directed_line_segment<R: Rng>(rng: &mut R) -> DirectedLineSegment {
        DirectedLineSegment::new(rand_coord(rng), rand_coord(rng))
    }

    #[test]
    fn iterator_reaches_end() {
        test_properties(DirectedLineSegment::new(Coord::new(0, 0), Coord::new(0, 0)));
        test_properties(DirectedLineSegment::new(Coord::new(0, 0), Coord::new(1, 1)));
        test_properties(DirectedLineSegment::new(Coord::new(0, 0), Coord::new(1, 0)));
        test_properties(DirectedLineSegment::new(Coord::new(0, 0), Coord::new(2, 1)));
        test_properties(DirectedLineSegment::new(
            Coord::new(1, -1),
            Coord::new(0, 0),
        ));
        test_properties(DirectedLineSegment::new(
            Coord::new(1, 100),
            Coord::new(0, 0),
        ));
        test_properties(DirectedLineSegment::new(
            Coord::new(100, 1),
            Coord::new(0, 0),
        ));
        const NUM_RAND_TESTS: usize = 10000;
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..NUM_RAND_TESTS {
            test_properties(rand_directed_line_segment(&mut rng));
        }
    }
}
