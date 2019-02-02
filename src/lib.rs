extern crate coord_2d;
use coord_2d::Axis;
pub use coord_2d::Coord;

#[derive(Debug, Clone, Copy)]
pub struct GeneralLineSegmentIter<S: StepsTrait> {
    steps: S,
    current_coord: Coord,
    remaining: usize,
}

pub type LineSegmentIter = GeneralLineSegmentIter<Steps>;
pub type LineSegmentIterCardinal = GeneralLineSegmentIter<StepsCardinal>;

pub trait TraverseTrait: Clone + Eq + private::Sealed {
    type Steps: StepsTrait;
    type Iter: Iterator<Item = Coord>;
    fn num_steps(&self) -> usize;
    fn steps(&self) -> Self::Steps;
    fn iter(&self) -> Self::Iter;
    fn line_segment(&self) -> LineSegment;
    fn start(&self) -> Coord {
        self.line_segment().start
    }
    fn end(&self) -> Coord {
        self.line_segment().end
    }
}

pub trait StepsTrait: Clone + Eq + private::Sealed {
    fn prev(&mut self) -> Coord;
    fn next(&mut self) -> Coord;
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Config {
    pub exclude_start: bool,
    pub exclude_end: bool,
}

impl Config {
    pub fn include_start(self) -> Self {
        Self {
            exclude_start: false,
            ..self
        }
    }
    pub fn include_end(self) -> Self {
        Self {
            exclude_end: false,
            ..self
        }
    }
    pub fn exclude_start(self) -> Self {
        Self {
            exclude_start: true,
            ..self
        }
    }
    pub fn extlude_end(self) -> Self {
        Self {
            exclude_end: true,
            ..self
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LineSegment {
    pub start: Coord,
    pub end: Coord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Traverse {
    pub line_segment: LineSegment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TraverseCardinal {
    pub line_segment: LineSegment,
}

impl LineSegment {
    pub fn new(start: Coord, end: Coord) -> Self {
        Self { start, end }
    }
    pub fn traverse(&self) -> Traverse {
        Traverse {
            line_segment: *self,
        }
    }
    pub fn traverse_cardinal(&self) -> TraverseCardinal {
        TraverseCardinal {
            line_segment: *self,
        }
    }
    pub fn delta(&self) -> Coord {
        self.end - self.start
    }
    pub fn iter(&self) -> LineSegmentIter {
        GeneralLineSegmentIter::new(self.traverse())
    }
    pub fn iter_cardinal(&self) -> LineSegmentIterCardinal {
        GeneralLineSegmentIter::new(self.traverse_cardinal())
    }
    pub fn steps(&self) -> Steps {
        Steps::new(self.delta())
    }
    pub fn steps_cardinal(&self) -> StepsCardinal {
        StepsCardinal::new(self.delta())
    }
    pub fn num_steps(&self) -> usize {
        let delta = self.delta();
        delta.x.abs().max(delta.y.abs()) as usize + 1
    }
    pub fn num_steps_cardinal(&self) -> usize {
        let delta = self.delta();
        delta.x.abs() as usize + delta.y.abs() as usize + 1
    }
}

impl TraverseTrait for Traverse {
    type Steps = Steps;
    type Iter = LineSegmentIter;
    fn num_steps(&self) -> usize {
        self.line_segment.num_steps()
    }
    fn steps(&self) -> Self::Steps {
        self.line_segment.steps()
    }
    fn iter(&self) -> Self::Iter {
        self.line_segment.iter()
    }
    fn line_segment(&self) -> LineSegment {
        self.line_segment
    }
}

impl TraverseTrait for TraverseCardinal {
    type Steps = StepsCardinal;
    type Iter = LineSegmentIterCardinal;
    fn num_steps(&self) -> usize {
        self.line_segment.num_steps_cardinal()
    }
    fn steps(&self) -> Self::Steps {
        self.line_segment.steps_cardinal()
    }
    fn iter(&self) -> LineSegmentIterCardinal {
        self.line_segment.iter_cardinal()
    }
    fn line_segment(&self) -> LineSegment {
        self.line_segment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Steps {
    major_axis: Axis,
    major_sign: i8,
    minor_sign: i8,
    accumulator: i64,
    major_delta_abs: u32,
    minor_delta_abs: u32,
}

impl Steps {
    pub fn new(delta: Coord) -> Self {
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
}

impl StepsTrait for Steps {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepsCardinal(Steps);

impl StepsCardinal {
    fn new(delta: Coord) -> Self {
        StepsCardinal(Steps::new(delta))
    }
}

impl StepsTrait for StepsCardinal {
    fn prev(&mut self) -> Coord {
        self.0.accumulator -= self.0.minor_delta_abs as i64;
        if self.0.accumulator
            <= (self.0.major_delta_abs as i64 / 2)
                - self.0.major_delta_abs as i64
                - self.0.minor_delta_abs as i64
        {
            self.0.accumulator += self.0.major_delta_abs as i64 + self.0.minor_delta_abs as i64;;
            Coord::new_axis(0, -self.0.minor_sign as i32, self.0.major_axis)
        } else {
            Coord::new_axis(-self.0.major_sign as i32, 0, self.0.major_axis)
        }
    }
    fn next(&mut self) -> Coord {
        self.0.accumulator += self.0.minor_delta_abs as i64;
        if self.0.accumulator > self.0.major_delta_abs as i64 / 2 {
            self.0.accumulator -= self.0.major_delta_abs as i64 + self.0.minor_delta_abs as i64;
            Coord::new_axis(0, self.0.minor_sign as i32, self.0.major_axis)
        } else {
            Coord::new_axis(self.0.major_sign as i32, 0, self.0.major_axis)
        }
    }
}

impl<S: StepsTrait> GeneralLineSegmentIter<S> {
    fn new<T: TraverseTrait<Steps = S>>(traverse: T) -> Self {
        let mut steps = traverse.steps();
        let backwards_step = steps.prev();
        let current_coord = traverse.start() + backwards_step;
        let remaining = traverse.num_steps();
        Self {
            steps,
            current_coord,
            remaining,
        }
    }
}

impl<S: StepsTrait> Iterator for GeneralLineSegmentIter<S> {
    type Item = Coord;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let step = self.steps.next();
        self.current_coord += step;
        self.remaining -= 1;
        Some(self.current_coord)
    }
}

impl IntoIterator for LineSegment {
    type Item = Coord;
    type IntoIter = LineSegmentIter;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

mod private {
    pub trait Sealed {}

    impl Sealed for super::Steps {}
    impl Sealed for super::StepsCardinal {}
    impl Sealed for super::Traverse {}
    impl Sealed for super::TraverseCardinal {}
}

#[cfg(test)]
mod test {
    extern crate rand;
    use self::rand::rngs::StdRng;
    use self::rand::{Rng, SeedableRng};
    use super::*;

    fn test_properties_gen<T>(traverse: T)
    where
        T: TraverseTrait + ::std::fmt::Debug,
        T::Steps: ::std::fmt::Debug,
    {
        let coords: Vec<_> = traverse.iter().collect();
        assert_eq!(coords.len(), traverse.num_steps());
        assert_eq!(*coords.first().unwrap(), traverse.start());
        assert_eq!(*coords.last().unwrap(), traverse.end());
        let mut steps = traverse.steps();
        for _ in 0..traverse.num_steps() {
            let before = steps.clone();
            steps.next();
            let mut after = steps.clone();
            after.prev();
            assert_eq!(
                before, after,
                "\n{:#?}\n{:#?}\n{:#?}",
                before, after, traverse
            );
        }
    }

    fn test_properties(line_segment: LineSegment) {
        test_properties_gen(line_segment.traverse());
        test_properties_gen(line_segment.traverse_cardinal());
    }

    fn rand_int<R: Rng>(rng: &mut R) -> i32 {
        const MAX: i32 = 100;
        rng.gen::<i32>() % MAX
    }

    fn rand_coord<R: Rng>(rng: &mut R) -> Coord {
        Coord::new(rand_int(rng), rand_int(rng))
    }

    fn rand_line_segment<R: Rng>(rng: &mut R) -> LineSegment {
        LineSegment::new(rand_coord(rng), rand_coord(rng))
    }

    #[test]
    fn all() {
        test_properties(LineSegment::new(Coord::new(0, 0), Coord::new(0, 0)));
        test_properties(LineSegment::new(Coord::new(0, 0), Coord::new(1, 1)));
        test_properties(LineSegment::new(Coord::new(0, 0), Coord::new(1, 0)));
        test_properties(LineSegment::new(Coord::new(0, 0), Coord::new(2, 1)));
        test_properties(LineSegment::new(Coord::new(1, -1), Coord::new(0, 0)));
        test_properties(LineSegment::new(Coord::new(1, 100), Coord::new(0, 0)));
        test_properties(LineSegment::new(Coord::new(100, 1), Coord::new(0, 0)));
        const NUM_RAND_TESTS: usize = 10000;
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..NUM_RAND_TESTS {
            test_properties(rand_line_segment(&mut rng));
        }
    }
}
