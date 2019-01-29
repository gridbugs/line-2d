extern crate coord_2d;
use coord_2d::Axis;
pub use coord_2d::Coord;

pub trait StepsTrait: Clone + Eq + private::Sealed {
    fn prev(&mut self) -> Coord;
    fn next(&mut self) -> Coord;
}

pub trait LineSegmentTrait: private::Sealed {
    type Steps: StepsTrait;
    fn num_steps(&self) -> usize;
    fn steps(&self) -> Self::Steps;
    fn iter(&self) -> LineSegmentIter<Self::Steps>;
    fn start(&self) -> Coord;
    fn end(&self) -> Coord;
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

impl LineSegment {
    pub fn new(start: Coord, end: Coord) -> Self {
        Self { start, end }
    }
    pub fn delta(&self) -> Coord {
        self.end - self.start
    }
    pub fn iter(&self) -> LineSegmentIter<Steps> {
        LineSegmentIter::new(*self)
    }
    pub fn iter_cardinal(&self) -> LineSegmentIter<StepsCardinal> {
        LineSegmentIter::new(LineSegmentCardinal(*self))
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
    pub fn cardinal(&self) -> LineSegmentCardinal {
        LineSegmentCardinal(*self)
    }
}

impl LineSegmentTrait for LineSegment {
    type Steps = Steps;
    fn num_steps(&self) -> usize {
        LineSegment::num_steps(self)
    }
    fn steps(&self) -> Self::Steps {
        LineSegment::steps(self)
    }
    fn iter(&self) -> LineSegmentIter<Self::Steps> {
        LineSegment::iter(self)
    }
    fn start(&self) -> Coord {
        self.start
    }
    fn end(&self) -> Coord {
        self.end
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LineSegmentCardinal(LineSegment);

impl LineSegmentCardinal {
    pub fn new(start: Coord, end: Coord) -> Self {
        LineSegmentCardinal(LineSegment::new(start, end))
    }
    pub fn line_segment(&self) -> LineSegment {
        self.0
    }
}

impl LineSegmentTrait for LineSegmentCardinal {
    type Steps = StepsCardinal;
    fn num_steps(&self) -> usize {
        LineSegment::num_steps_cardinal(&self.0)
    }
    fn steps(&self) -> Self::Steps {
        LineSegment::steps_cardinal(&self.0)
    }
    fn iter(&self) -> LineSegmentIter<Self::Steps> {
        LineSegment::iter_cardinal(&self.0)
    }
    fn start(&self) -> Coord {
        self.0.start
    }
    fn end(&self) -> Coord {
        self.0.end
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

#[derive(Debug, Clone, Copy)]
pub struct LineSegmentIter<S: StepsTrait> {
    steps: S,
    current_coord: Coord,
    remaining: usize,
}

impl<S: StepsTrait> LineSegmentIter<S> {
    fn new<L: LineSegmentTrait<Steps = S>>(line_segment: L) -> Self {
        let mut steps = line_segment.steps();
        let backwards_step = steps.prev();
        let current_coord = line_segment.start() + backwards_step;
        let remaining = line_segment.num_steps();
        Self {
            steps,
            current_coord,
            remaining,
        }
    }
}

impl<S: StepsTrait> Iterator for LineSegmentIter<S> {
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
    type IntoIter = LineSegmentIter<Steps>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for LineSegmentCardinal {
    type Item = Coord;
    type IntoIter = LineSegmentIter<StepsCardinal>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub fn line_segment(start: Coord, end: Coord) -> LineSegment {
    LineSegment::new(start, end)
}

pub fn line_segment_cardinal(start: Coord, end: Coord) -> LineSegmentCardinal {
    LineSegmentCardinal::new(start, end)
}

mod private {
    pub trait Sealed {}

    impl Sealed for super::LineSegment {}
    impl Sealed for super::LineSegmentCardinal {}
    impl Sealed for super::Steps {}
    impl Sealed for super::StepsCardinal {}
}

#[cfg(test)]
mod test {
    extern crate rand;
    use self::rand::rngs::StdRng;
    use self::rand::{Rng, SeedableRng};
    use super::*;

    fn test_properties_gen<L>(line_segment: L)
    where
        L: LineSegmentTrait + ::std::fmt::Debug,
        L::Steps: ::std::fmt::Debug,
    {
        let coords: Vec<_> = line_segment.iter().collect();
        assert_eq!(coords.len(), line_segment.num_steps());
        assert_eq!(*coords.first().unwrap(), line_segment.start());
        assert_eq!(*coords.last().unwrap(), line_segment.end());
        let mut steps = line_segment.steps();
        for _ in 0..line_segment.num_steps() {
            let before = steps.clone();
            steps.next();
            let mut after = steps.clone();
            after.prev();
            assert_eq!(
                before, after,
                "\n{:#?}\n{:#?}\n{:#?}",
                before, after, line_segment
            );
        }
    }

    fn test_properties(line_segment: LineSegment) {
        test_properties_gen(line_segment);
        test_properties_gen(line_segment.cardinal());
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
    fn iterator_reaches_end() {
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
