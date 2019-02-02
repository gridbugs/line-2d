#[cfg(feature = "serialize")]
#[macro_use]
extern crate serde;

extern crate coord_2d;
use coord_2d::Axis;
pub use coord_2d::Coord;

pub trait StepsTrait: Clone + Eq + private::Sealed {
    fn prev(&mut self) -> Coord;
    fn next(&mut self) -> Coord;
    fn prev_copy(&self) -> (Coord, Self) {
        let mut s = self.clone();
        (s.prev(), s)
    }
    fn next_copy(&self) -> (Coord, Self) {
        let mut s = self.clone();
        (s.next(), s)
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Steps {
    major_x: i8,
    major_y: i8,
    minor_x: i8,
    minor_y: i8,
    accumulator: i64,
    major_delta_abs: u32,
    minor_delta_abs: u32,
}

impl Steps {
    fn new_common<F: Fn(i8, i8, Axis) -> (Coord, Coord)>(delta: Coord, f: F) -> Self {
        let (major_axis, minor_axis) = if delta.x.abs() > delta.y.abs() {
            (Axis::X, Axis::Y)
        } else {
            (Axis::Y, Axis::X)
        };
        let major_sign = if delta.get(major_axis) < 0 { -1 } else { 1 };
        let minor_sign = if delta.get(minor_axis) < 0 { -1 } else { 1 };
        let (major_coord, minor_coord) = f(major_sign, minor_sign, major_axis);
        let (major_delta_abs, minor_delta_abs) = if delta == Coord::new(0, 0) {
            // this prevents traversal from looping forever
            (1, 1)
        } else {
            (
                delta.get(major_axis).abs() as u32,
                delta.get(minor_axis).abs() as u32,
            )
        };
        let accumulator = 0;
        let major_x = major_coord.x as i8;
        let major_y = major_coord.y as i8;
        let minor_x = minor_coord.x as i8;
        let minor_y = minor_coord.y as i8;
        Self {
            major_x,
            major_y,
            minor_x,
            minor_y,
            accumulator,
            major_delta_abs,
            minor_delta_abs,
        }
    }
    pub fn new(delta: Coord) -> Self {
        Self::new_common(delta, |major_sign, minor_sign, major_axis| {
            let major_coord = Coord::new_axis(major_sign as i32, 0, major_axis);
            let minor_coord = Coord::new_axis(major_sign as i32, minor_sign as i32, major_axis);
            (major_coord, minor_coord)
        })
    }
}

impl StepsTrait for Steps {
    fn prev(&mut self) -> Coord {
        self.accumulator -= self.minor_delta_abs as i64;
        if self.accumulator <= (self.major_delta_abs as i64 / 2) - self.major_delta_abs as i64 {
            self.accumulator += self.major_delta_abs as i64;
            Coord::new(-self.minor_x as i32, -self.minor_y as i32)
        } else {
            Coord::new(-self.major_x as i32, -self.major_y as i32)
        }
    }
    fn next(&mut self) -> Coord {
        self.accumulator += self.minor_delta_abs as i64;
        if self.accumulator > self.major_delta_abs as i64 / 2 {
            self.accumulator -= self.major_delta_abs as i64;
            Coord::new(self.minor_x as i32, self.minor_y as i32)
        } else {
            Coord::new(self.major_x as i32, self.major_y as i32)
        }
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepsCardinal(Steps);

impl StepsCardinal {
    fn new(delta: Coord) -> Self {
        let steps = Steps::new_common(delta, |major_sign, minor_sign, major_axis| {
            let major_coord = Coord::new_axis(major_sign as i32, 0, major_axis);
            let minor_coord = Coord::new_axis(0, minor_sign as i32, major_axis);
            (major_coord, minor_coord)
        });
        StepsCardinal(steps)
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
            Coord::new(-self.0.minor_x as i32, -self.0.minor_y as i32)
        } else {
            Coord::new(-self.0.major_x as i32, -self.0.major_y as i32)
        }
    }
    fn next(&mut self) -> Coord {
        self.0.accumulator += self.0.minor_delta_abs as i64;
        if self.0.accumulator > self.0.major_delta_abs as i64 / 2 {
            self.0.accumulator -= self.0.major_delta_abs as i64 + self.0.minor_delta_abs as i64;
            Coord::new(self.0.minor_x as i32, self.0.minor_y as i32)
        } else {
            Coord::new(self.0.major_x as i32, self.0.major_y as i32)
        }
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LineSegment {
    pub start: Coord,
    pub end: Coord,
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
    pub fn iter_config(&self, config: Config) -> LineSegmentIter {
        GeneralLineSegmentIter::new_config(self.traverse(), config)
    }
    pub fn iter_cardinal_config(&self, config: Config) -> LineSegmentIterCardinal {
        GeneralLineSegmentIter::new_config(self.traverse_cardinal(), config)
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
    pub fn reverse(&self) -> Self {
        Self {
            start: self.end,
            end: self.start,
        }
    }
    pub fn try_infinite(&self) -> Result<InfiniteLineSegment, ZeroLengthDelta> {
        InfiniteLineSegment::try_new(self.start, self.delta())
    }
    pub fn infinite(&self) -> InfiniteLineSegment {
        self.try_infinite().unwrap()
    }
}

pub trait TraverseTrait: Clone + Eq + private::Sealed {
    type Steps: StepsTrait;
    type Iter: Iterator<Item = Coord>;
    fn num_steps(&self) -> usize;
    fn steps(&self) -> Self::Steps;
    fn iter(&self) -> Self::Iter;
    fn iter_config(&self, config: Config) -> Self::Iter;
    fn line_segment(&self) -> LineSegment;
    fn start(&self) -> Coord {
        self.line_segment().start
    }
    fn end(&self) -> Coord {
        self.line_segment().end
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Traverse {
    pub line_segment: LineSegment,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TraverseCardinal {
    pub line_segment: LineSegment,
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
    fn iter_config(&self, config: Config) -> Self::Iter {
        self.line_segment.iter_config(config)
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
    fn iter_config(&self, config: Config) -> LineSegmentIterCardinal {
        self.line_segment.iter_cardinal_config(config)
    }
    fn line_segment(&self) -> LineSegment {
        self.line_segment
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Default, Debug, Clone, Copy)]
pub struct Config {
    pub exclude_start: bool,
    pub exclude_end: bool,
}

impl Config {
    pub fn new() -> Self {
        Default::default()
    }
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
    pub fn exclude_end(self) -> Self {
        Self {
            exclude_end: true,
            ..self
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GeneralLineSegmentIter<S: StepsTrait> {
    steps: S,
    current_coord: Coord,
    remaining: usize,
}

pub type LineSegmentIter = GeneralLineSegmentIter<Steps>;
pub type LineSegmentIterCardinal = GeneralLineSegmentIter<StepsCardinal>;

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

    fn new_config<T: TraverseTrait<Steps = S>>(traverse: T, config: Config) -> Self {
        let mut iter = Self::new(traverse);
        if config.exclude_end {
            iter.remaining -= 1;
        }
        if config.exclude_start {
            iter.next();
        }
        iter
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

impl IntoIterator for Traverse {
    type Item = Coord;
    type IntoIter = LineSegmentIter;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for TraverseCardinal {
    type Item = Coord;
    type IntoIter = LineSegmentIterCardinal;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InfiniteLineSegment {
    line_segment: LineSegment,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct ZeroLengthDelta;

impl InfiniteLineSegment {
    pub fn try_new(start: Coord, delta: Coord) -> Result<Self, ZeroLengthDelta> {
        if delta == Coord::new(0, 0) {
            Err(ZeroLengthDelta)
        } else {
            Ok(Self {
                line_segment: LineSegment::new(start, start + delta),
            })
        }
    }
    pub fn new(start: Coord, delta: Coord) -> Self {
        Self::try_new(start, delta).unwrap()
    }
    pub fn start(&self) -> Coord {
        self.line_segment.start
    }
    pub fn delta(&self) -> Coord {
        self.line_segment.delta()
    }
    pub fn reverse(&self) -> Self {
        let start = self.start();
        let end = start - self.delta();
        Self {
            line_segment: LineSegment::new(start, end),
        }
    }
    pub fn traverse(&self) -> InfiniteTraverse {
        InfiniteTraverse {
            line_segment: *self,
        }
    }
    pub fn traverse_cardinal(&self) -> InfiniteTraverseCardinal {
        InfiniteTraverseCardinal {
            line_segment: *self,
        }
    }
    pub fn iter(&self) -> InfiniteLineSegmentIter {
        GeneralInfiniteLineSegmentIter::new(self.traverse())
    }
    pub fn iter_config(&self, config: InfiniteConfig) -> InfiniteLineSegmentIter {
        if config.exclude_start {
            GeneralInfiniteLineSegmentIter::new_exclude_start(self.traverse())
        } else {
            self.iter()
        }
    }
    pub fn iter_cardinal(&self) -> InfiniteLineSegmentIterCardinal {
        GeneralInfiniteLineSegmentIter::new(self.traverse_cardinal())
    }
    pub fn iter_cardinal_config(&self, config: InfiniteConfig) -> InfiniteLineSegmentIterCardinal {
        if config.exclude_start {
            GeneralInfiniteLineSegmentIter::new_exclude_start(self.traverse_cardinal())
        } else {
            self.iter_cardinal()
        }
    }
}

pub trait InfiniteTraverseTrait: Clone + Eq + private::Sealed {
    type Steps: StepsTrait;
    type Iter: Iterator<Item = Coord>;
    fn steps(&self) -> Self::Steps;
    fn iter(&self) -> Self::Iter;
    fn iter_config(&self, config: InfiniteConfig) -> Self::Iter;
    fn start(&self) -> Coord;
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InfiniteTraverse {
    line_segment: InfiniteLineSegment,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InfiniteTraverseCardinal {
    line_segment: InfiniteLineSegment,
}

impl InfiniteTraverseTrait for InfiniteTraverse {
    type Steps = Steps;
    type Iter = InfiniteLineSegmentIter;
    fn steps(&self) -> Self::Steps {
        Steps::new(self.line_segment.delta())
    }
    fn iter(&self) -> Self::Iter {
        self.line_segment.iter()
    }
    fn iter_config(&self, config: InfiniteConfig) -> Self::Iter {
        self.line_segment.iter_config(config)
    }
    fn start(&self) -> Coord {
        self.line_segment.start()
    }
}

impl InfiniteTraverseTrait for InfiniteTraverseCardinal {
    type Steps = StepsCardinal;
    type Iter = InfiniteLineSegmentIterCardinal;
    fn steps(&self) -> Self::Steps {
        StepsCardinal::new(self.line_segment.delta())
    }
    fn iter(&self) -> Self::Iter {
        self.line_segment.iter_cardinal()
    }
    fn iter_config(&self, config: InfiniteConfig) -> Self::Iter {
        self.line_segment.iter_cardinal_config(config)
    }
    fn start(&self) -> Coord {
        self.line_segment.start()
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Default, Debug, Clone, Copy)]
pub struct InfiniteConfig {
    pub exclude_start: bool,
}

impl InfiniteConfig {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn include_start(self) -> Self {
        Self {
            exclude_start: false,
            ..self
        }
    }
    pub fn exclude_start(self) -> Self {
        Self {
            exclude_start: true,
            ..self
        }
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct GeneralInfiniteLineSegmentIter<S: StepsTrait> {
    steps: S,
    current_coord: Coord,
}

pub type InfiniteLineSegmentIter = GeneralInfiniteLineSegmentIter<Steps>;
pub type InfiniteLineSegmentIterCardinal = GeneralInfiniteLineSegmentIter<StepsCardinal>;

impl<S: StepsTrait> GeneralInfiniteLineSegmentIter<S> {
    fn new<T: InfiniteTraverseTrait<Steps = S>>(traverse: T) -> Self {
        let mut steps = traverse.steps();
        let backwards_step = steps.prev();
        let current_coord = traverse.start() + backwards_step;
        Self {
            steps,
            current_coord,
        }
    }

    fn new_exclude_start<T: InfiniteTraverseTrait<Steps = S>>(traverse: T) -> Self {
        let steps = traverse.steps();
        let current_coord = traverse.start();
        Self {
            steps,
            current_coord,
        }
    }
}

impl<S: StepsTrait> Iterator for GeneralInfiniteLineSegmentIter<S> {
    type Item = Coord;
    fn next(&mut self) -> Option<Self::Item> {
        let (delta, next_steps) = self.steps.next_copy();
        if let Some(next_coord) = self.current_coord.checked_add(delta) {
            self.current_coord = next_coord;
            self.steps = next_steps;
            Some(next_coord)
        } else {
            None
        }
    }
}

impl IntoIterator for InfiniteLineSegment {
    type Item = Coord;
    type IntoIter = InfiniteLineSegmentIter;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for InfiniteTraverse {
    type Item = Coord;
    type IntoIter = InfiniteLineSegmentIter;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for InfiniteTraverseCardinal {
    type Item = Coord;
    type IntoIter = InfiniteLineSegmentIterCardinal;
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
    impl Sealed for super::InfiniteTraverse {}
    impl Sealed for super::InfiniteTraverseCardinal {}
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
        let orig_coords = coords;
        let coords: Vec<_> = traverse
            .iter_config(Config::new().exclude_start().exclude_end())
            .collect();
        assert_eq!(coords.len(), traverse.num_steps().max(2) - 2);
        if let Some(&coord) = coords.first() {
            assert_eq!(coord, orig_coords[1]);
        }
        if let Some(&coord) = coords.last() {
            assert_eq!(coord, orig_coords[orig_coords.len() - 2]);
        }
        let coords: Vec<_> = traverse
            .iter_config(Config::new().exclude_start())
            .collect();
        assert_eq!(coords.len(), traverse.num_steps().max(1) - 1);
        if let Some(&coord) = coords.first() {
            assert_eq!(coord, orig_coords[1]);
        }
        if let Some(&coord) = coords.last() {
            assert_eq!(coord, orig_coords[orig_coords.len() - 1]);
        }
        let coords: Vec<_> = traverse.iter_config(Config::new().exclude_end()).collect();
        assert_eq!(coords.len(), traverse.num_steps().max(1) - 1);
        if let Some(&coord) = coords.first() {
            assert_eq!(coord, orig_coords[0]);
        }
        if let Some(&coord) = coords.last() {
            assert_eq!(coord, orig_coords[orig_coords.len() - 2]);
        }
    }

    fn compare_finite_with_infinite<F: TraverseTrait, I: InfiniteTraverseTrait>(f: F, i: I) {
        f.iter().zip(i.iter()).for_each(|(f, i)| {
            assert_eq!(f, i);
        });
        f.iter_config(Config::new().exclude_start())
            .zip(i.iter_config(InfiniteConfig::new().exclude_start()))
            .for_each(|(f, i)| {
                assert_eq!(f, i);
            });
    }

    fn test_properties(line_segment: LineSegment) {
        test_properties_gen(line_segment.traverse());
        test_properties_gen(line_segment.traverse_cardinal());
        if let Ok(infinite) = line_segment.try_infinite() {
            compare_finite_with_infinite(line_segment.traverse(), infinite.traverse());
            compare_finite_with_infinite(
                line_segment.traverse_cardinal(),
                infinite.traverse_cardinal(),
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
